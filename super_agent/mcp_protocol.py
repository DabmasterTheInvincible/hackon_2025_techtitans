import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import httpx

import redis.asyncio as redis
from kafka import KafkaProducer

from event_router import EventRouter, EventType, MarketplaceEvent

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class MCPPacket:
    """Model Context Protocol packet structure"""
    event_id: str
    trace_id: str
    timestamp: datetime
    event_type: EventType
    raw_data: Dict[str, Any]
    context: Dict[str, Any]
    sub_agent_results: Dict[str, Any]
    risk_scores: Dict[str, float]
    final_risk_level: Optional[RiskLevel] = None
    investigation_brief: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['event_type'] = self.event_type.value
        if self.final_risk_level:
            result['final_risk_level'] = self.final_risk_level.value
        result['timestamp'] = self.timestamp.isoformat()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCPPacket':
        data['event_type'] = EventType(data['event_type'])
        if data.get('final_risk_level'):
            data['final_risk_level'] = RiskLevel(data['final_risk_level'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

class TSCCSuperAgent:
    def __init__(self):
        self.redis_client = None
        self.kafka_producer = None
        self.event_router = EventRouter()
        self.metrics = {
            "events_processed": 0,
            "high_risk_alerts": 0,
            "avg_processing_time": 0.0
        }

    async def initialize(self):
        try:
            self.redis_client = redis.from_url("redis://localhost:6379")
            await self.redis_client.ping()
            logger.info("Redis connection established")

            self.kafka_producer = KafkaProducer(
                #bootstrap_servers=['kafka:9092'],
                bootstrap_servers=["localhost:9092"],
                value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
                acks='all',
                retries=3,
                max_in_flight_requests_per_connection=1
            )
            logger.info("Kafka producer initialized")

        except Exception as e:
            logger.error(f"Failed to initialize super agent: {e}")
            raise

    async def cleanup(self):
        try:
            if self.redis_client:
                await self.redis_client.close()
            if self.kafka_producer:
                self.kafka_producer.close()
            logger.info("Connections cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def create_mcp_packet(self, event: MarketplaceEvent) -> MCPPacket:
        event_id = str(uuid.uuid4())
        trace_id = str(uuid.uuid4())
        context = await self._get_context_data(event)
        return MCPPacket(
            event_id=event_id,
            trace_id=trace_id,
            timestamp=datetime.now(),
            event_type=EventType(event.event_type),
            raw_data=event.content,
            context=context,
            sub_agent_results={},
            risk_scores={}
        )

    async def route_to_sub_agents(self, mcp_packet: MCPPacket):
        try:
            relevant_agents = self.event_router.get_routing_rules(mcp_packet.event_type)
            for agent_name in relevant_agents:
                topic = self.event_router.get_agent_topic(agent_name)
                message = {
                    "mcp_packet": mcp_packet.to_dict(),
                    "agent_name": agent_name,
                    "routing_timestamp": datetime.now().isoformat()
                }
                future = self.kafka_producer.send(topic, message)
                future.get(timeout=10)
                logger.info(f"Sent MCP packet {mcp_packet.trace_id} to {agent_name}")
            self.kafka_producer.flush()
        except Exception as e:
            logger.error(f"Error routing MCP packet {mcp_packet.trace_id}: {e}")
            raise

    async def aggregate_results(self, trace_id: str, timeout: int = 100) -> MCPPacket:
        start_time = asyncio.get_event_loop().time()
        cached_result = None
        while asyncio.get_event_loop().time() - start_time < timeout:
            try:
                result_key = f"mcp_results:{trace_id}"
                cached_result = await self.redis_client.get(result_key)
                if cached_result:
                    result_data = json.loads(cached_result)
                    mcp_packet = MCPPacket.from_dict(result_data)
                    if self._all_agents_responded(mcp_packet):
                        logger.info(f"All agents responded for trace_id: {trace_id}")
                        return mcp_packet
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error checking results for trace_id {trace_id}: {e}")
                await asyncio.sleep(1)
        logger.warning(f"Timeout waiting for all sub-agents for trace_id: {trace_id}")
        if cached_result:
            return MCPPacket.from_dict(json.loads(cached_result))
        raise TimeoutError(f"Failed to get results from sub-agents for trace_id: {trace_id}")

    def _all_agents_responded(self, mcp_packet: MCPPacket) -> bool:
        expected_agents = set(self.event_router.get_routing_rules(mcp_packet.event_type))
        responded_agents = set(mcp_packet.sub_agent_results.keys())
        return expected_agents.issubset(responded_agents)

    async def calculate_final_risk(self, mcp_packet: MCPPacket) -> RiskLevel:
        if not mcp_packet.risk_scores:
            return RiskLevel.LOW
        weights = {"fraud_detection": 0.4, "counterfeit_detection": 0.3, "review_spam": 0.2, "return_anomaly": 0.1}
        weighted_score = sum(score * weights.get(agent, 0) for agent, score in mcp_packet.risk_scores.items())
        total_weight = sum(weights.get(agent, 0) for agent in mcp_packet.risk_scores)
        final_score = weighted_score / total_weight if total_weight else 0
        if final_score >= 0.8:
            return RiskLevel.CRITICAL
        elif final_score >= 0.6:
            return RiskLevel.HIGH
        elif final_score >= 0.3:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    async def generate_investigation_brief(self, mcp_packet: MCPPacket) -> str:
        if mcp_packet.final_risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            try:
                prompt = self._create_investigation_prompt(mcp_packet)
                async with httpx.AsyncClient() as client:
                    response = await client.post("http://ollama:11434/api/generate", json={
                        "model": "gemma",
                        "prompt": prompt,
                        "stream": False
                    })
                    if response.status_code == 200:
                        return response.json().get("response", "[No response from Gemma]").strip()
                    else:
                        logger.warning(f"Gemma returned status {response.status_code}")
            except Exception as e:
                logger.error(f"Error generating brief via Gemma: {e}")
                max_score = max(mcp_packet.risk_scores.values()) if mcp_packet.risk_scores else 0
                return f"Auto-generated brief: High risk event detected with score {max_score:.2f}. Manual investigation required."
        return "Low risk event - no investigation brief generated"

    def _create_investigation_prompt(self, mcp_packet: MCPPacket) -> str:
        return f"""
        Analyze this marketplace event and create a concise investigation brief:
        Event Type: {mcp_packet.event_type.value}
        Event ID: {mcp_packet.event_id}
        Timestamp: {mcp_packet.timestamp}
        Final Risk Level: {mcp_packet.final_risk_level.value if mcp_packet.final_risk_level else 'Unknown'}
        Risk Scores:
        {json.dumps(mcp_packet.risk_scores, indent=2)}
        Sub-Agent Results Summary:
        {self._summarize_agent_results(mcp_packet.sub_agent_results)}
        Event Context:
        {json.dumps(mcp_packet.context, indent=2)}
        Please provide:
        1. Risk summary (2-3 sentences)
        2. Key indicators that triggered alerts
        3. Recommended next actions
        4. Priority level justification
        Keep the brief under 200 words and focus on actionable insights.
        """

    def _summarize_agent_results(self, results: Dict[str, Any]) -> str:
        summary = {}
        for agent, result in results.items():
            if isinstance(result, dict):
                summary[agent] = {
                    "status": result.get("status", "unknown"),
                    "key_findings": result.get("key_findings", [])[:3]
                }
        return json.dumps(summary, indent=2)

    async def publish_high_priority_alert(self, mcp_packet: MCPPacket):
        try:
            alert_message = {
                "trace_id": mcp_packet.trace_id,
                "event_id": mcp_packet.event_id,
                "risk_level": mcp_packet.final_risk_level.value,
                "investigation_brief": mcp_packet.investigation_brief,
                "timestamp": datetime.now().isoformat(),
                "event_type": mcp_packet.event_type.value,
                "risk_scores": mcp_packet.risk_scores
            }
            future = self.kafka_producer.send('high-priority-alerts', alert_message)
            future.get(timeout=10)
            self.metrics["high_risk_alerts"] += 1
            logger.info(f"Published high priority alert for trace_id: {mcp_packet.trace_id}")
        except Exception as e:
            logger.error(f"Error publishing high priority alert: {e}")

    async def _get_context_data(self, event: MarketplaceEvent) -> Dict[str, Any]:
        context = {"timestamp": datetime.now().isoformat(), "event_metadata": event.metadata}
        try:
            if event.seller_id:
                seller_key = f"seller_history:{event.seller_id}"
                seller_data = await self.redis_client.get(seller_key)
                if seller_data:
                    context["seller_history"] = json.loads(seller_data)
            if event.customer_id:
                customer_key = f"customer_history:{event.customer_id}"
                customer_data = await self.redis_client.get(customer_key)
                if customer_data:
                    context["customer_history"] = json.loads(customer_data)
            if event.product_id:
                product_key = f"product_history:{event.product_id}"
                product_data = await self.redis_client.get(product_key)
                if product_data:
                    context["product_history"] = json.loads(product_data)
        except Exception as e:
            logger.error(f"Error getting context data: {e}")
        return context

    async def test_redis_connection(self) -> bool:
        try:
            await self.redis_client.ping()
            return True
        except Exception:
            return False

    def test_kafka_connection(self) -> bool:
        try:
            return self.kafka_producer is not None
        except Exception:
            return False

    async def get_metrics(self) -> Dict[str, Any]:
        try:
            self.metrics["events_processed"] += 1
            redis_info = await self.redis_client.info()
            return {
                "agent_metrics": self.metrics,
                "redis_info": {
                    "connected_clients": redis_info.get("connected_clients", 0),
                    "used_memory": redis_info.get("used_memory_human", "unknown"),
                    "uptime": redis_info.get("uptime_in_seconds", 0)
                },
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
