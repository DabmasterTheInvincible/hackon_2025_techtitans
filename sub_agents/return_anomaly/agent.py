#!/usr/bin/env python3
"""
Return Anomaly Detection Sub-Agent for TSCC
Processes MCP packets for return anomaly detection using MCP protocol
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, List
from dataclasses import dataclass
import os
import joblib
import redis.asyncio as redis
import numpy as np

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, CallToolResult, CallToolRequest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ReturnAnomalyResult:
    anomaly_score: float
    indicators: List[str]
    confidence: float
    model_version: str
    processing_time: float

class ReturnAnomalyAgent:
    def __init__(self):
        self.agent_name = "return_anomaly"
        self.redis_client = None
        self.model = None
        self.model_path = "models/return_anomaly_ensemble_model.pkl"
        self.anomaly_thresholds = {
                    "risk_score": 0.7,
                    "return_rate": 0.5,
                    "frequent_returns": 5,
                    "same_day_return_hours": 24,
                    "high_value_return": 2,
                    "category_abuse": 1,
                    "suspicious_reason": 1
                }

    async def initialize(self):
        self.redis_client = redis.from_url("redis://localhost:6379")
        self.model = joblib.load(self.model_path)
        logger.info("Model loaded successfully.")

    def preprocess(self, raw_data: Dict[str, Any], customer_history: Dict[str, Any]) -> np.ndarray:
        try:
            price = float(raw_data.get("price", 0))
            avg_order_value = float(customer_history.get("avg_order_value", 1))
            return_rate = float(customer_history.get("return_rate", 0))
            total_returns = int(customer_history.get("total_returns", 0))
            returns_last_30 = int(customer_history.get("returns_last_30_days", 0))

            order_ts = raw_data.get("order_timestamp")
            return_ts = raw_data.get("return_timestamp")
            hours_to_return = 999.0
            if order_ts and return_ts:
                order_dt = datetime.fromisoformat(order_ts.replace("Z", "+00:00"))
                return_dt = datetime.fromisoformat(return_ts.replace("Z", "+00:00"))
                hours_to_return = (return_dt - order_dt).total_seconds() / 3600

            price_to_avg_ratio = price / avg_order_value if avg_order_value > 0 else 0

            category = raw_data.get("category", "")
            frequent_categories = customer_history.get("frequent_return_categories", [])
            in_frequent_return_category = int(category in frequent_categories)

            return_reason = raw_data.get("return_reason", "").lower()
            suspicious_reason = int(any(x in return_reason for x in ["changed mind", "found cheaper", "not as described"]))

            features = np.array([[price, avg_order_value, return_rate, total_returns,
                                  returns_last_30, hours_to_return, price_to_avg_ratio,
                                  in_frequent_return_category, suspicious_reason]])
            return features
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            return np.zeros((1, 9))

    def detect_anomaly(self, mcp_packet: Dict[str, Any]) -> ReturnAnomalyResult:
        start_time = datetime.now()
        raw_data = mcp_packet.get("raw_data", {})
        context = mcp_packet.get("context", {})

        features = self.preprocess(raw_data, context)
        prediction = self.model.predict(features)[0]
        proba = self.model.predict_proba(features)[0][1]  # probability of being anomaly

        indicators = []
        if proba > 0.7: indicators.append("high_risk_score")
        if features[0][2] > 0.5: indicators.append("high_return_rate")
        if features[0][3] > 5: indicators.append("frequent_returns")
        if features[0][5] < 24: indicators.append("same_day_return")
        if features[0][6] > 2: indicators.append("high_value_return")
        if features[0][7] == 1: indicators.append("category_abuse")
        if features[0][8] == 1: indicators.append("suspicious_reason")

        return ReturnAnomalyResult(
            anomaly_score=proba,
            indicators=indicators,
            confidence=0.9,
            model_version="ensemble_v1",
            processing_time=(datetime.now() - start_time).total_seconds()
        )

    async def process_mcp_packet(self, mcp_packet: Dict[str, Any]):
        trace_id = mcp_packet.get("trace_id")
        result = self.detect_anomaly(mcp_packet)

        mcp_packet["sub_agent_results"][self.agent_name] = {
            "anomaly_score": result.anomaly_score,
            "indicators": result.indicators,
            "confidence": result.confidence,
            "model_version": result.model_version,
            "processing_time": result.processing_time,
            "processed_at": datetime.now().isoformat()
        }

        mcp_packet["risk_scores"][self.agent_name] = result.anomaly_score

        await self.redis_client.set(
            f"mcp_results:{trace_id}",
            json.dumps(mcp_packet, default=str),
            ex=3600
        )
        logger.info(f"Processed return anomaly for trace_id: {trace_id}, score: {result.anomaly_score:.3f}")

    async def run_kafka_consumer(self):
        """Run Kafka consumer for processing events"""
        consumer = KafkaConsumer(
            'return-anomaly-topic',
            #bootstrap_servers=['kafka:9092'],
            bootstrap_servers=['localhost:9092'],
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            group_id='return-anomaly-group',
            auto_offset_reset='latest'
        )

        logger.info("Started Kafka consumer for return anomaly detection")

        for message in consumer:
            try:
                data = message.value
                mcp_packet = data.get("mcp_packet")

                if mcp_packet:
                    await self.process_mcp_packet(mcp_packet)

            except Exception as e:
                logger.error(f"Error processing Kafka message: {e}")

# MCP Server Implementation
server = Server("return-anomaly-agent")

# Global agent instance
return_agent = ReturnAnomalyAgent()

@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available return anomaly detection tools."""
    return [
        Tool(
            name="analyze_return_anomaly",
            description="Analyze return patterns for anomaly detection",
            inputSchema={
                "type": "object",
                "properties": {
                    "event_data": {
                        "type": "object",
                        "description": "Return event data including timestamps and details"
                    },
                    "context": {
                        "type": "object",
                        "description": "Customer history and additional context"
                    }
                },
                "required": ["event_data"]
            }
        ),
        Tool(
            name="get_return_patterns",
            description="Get common return anomaly patterns and indicators",
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern_type": {
                        "type": "string",
                        "description": "Type of return pattern to retrieve (common, temporal, behavioral)"
                    }
                }
            }
        ),
        Tool(
            name="update_anomaly_thresholds",
            description="Update anomaly detection thresholds",
            inputSchema={
                "type": "object",
                "properties": {
                    "thresholds": {
                        "type": "object",
                        "description": "New threshold values for anomaly detection"
                    }
                },
                "required": ["thresholds"]
            }
        ),
        Tool(
            name="get_return_statistics",
            description="Get return statistics and trends",
            inputSchema={
                "type": "object",
                "properties": {
                    "time_period": {
                        "type": "string",
                        "description": "Time period for statistics (daily, weekly, monthly)"
                    },
                    "category": {
                        "type": "string",
                        "description": "Product category filter"
                    }
                }
            }
        )
    ]

@server.call_tool()
async def call_tool(request: CallToolRequest) -> CallToolResult:
    """Handle tool calls for return anomaly detection."""
    
    if request.name == "analyze_return_anomaly":
        event_data = request.arguments.get("event_data", {})
        context = request.arguments.get("context", {})
        
        # Create mock MCP packet for analysis
        mcp_packet = {
            "raw_data": event_data,
            "context": context,
            "timestamp": datetime.now().isoformat(),
            "sub_agent_results": {},
            "risk_scores": {}
        }
        
        result = return_agent.detect_anomaly(mcp_packet)
        
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps({
                    "anomaly_score": result.anomaly_score,
                    "indicators": result.indicators,
                    "confidence": result.confidence,
                    "model_version": result.model_version,
                    "processing_time": result.processing_time
                }, indent=2)
            )]
        )
    
    elif request.name == "get_return_patterns":
        pattern_type = request.arguments.get("pattern_type", "common")
        
        patterns = {
            "common": [
                "High return rate customers (>50%)",
                "Frequent returners (>5 returns)",
                "Same-day returns after purchase",
                "Repeated category returns",
                "High-value item returns"
            ],
            "temporal": [
                "Returns within 24 hours of purchase",
                "Returns during specific time periods",
                "Seasonal return patterns",
                "Weekend vs weekday return behavior"
            ],
            "behavioral": [
                "Suspicious return reasons",
                "Multiple returns in short timeframe",
                "Returns of items significantly above average order value",
                "Returns following specific purchase patterns"
            ]
        }
        
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps({
                    "pattern_type": pattern_type,
                    "patterns": patterns.get(pattern_type, patterns["common"])
                }, indent=2)
            )]
        )
    
    elif request.name == "update_anomaly_thresholds":
        thresholds = request.arguments.get("thresholds", {})
        
        # Update thresholds with validation
        valid_thresholds = {}
        for key, value in thresholds.items():
            if key in return_agent.anomaly_thresholds:
                try:
                    valid_thresholds[key] = float(value)
                    return_agent.anomaly_thresholds[key] = float(value)
                except ValueError:
                    continue
        
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps({
                    "status": "success",
                    "updated_thresholds": valid_thresholds,
                    "current_thresholds": return_agent.anomaly_thresholds,
                    "timestamp": datetime.now().isoformat()
                }, indent=2)
            )]
        )
    
    elif request.name == "get_return_statistics":
        time_period = request.arguments.get("time_period", "monthly")
        category = request.arguments.get("category", "all")
        
        # Mock statistics (in real implementation, would query actual data)
        statistics = {
            "time_period": time_period,
            "category": category,
            "total_returns": 1245,
            "return_rate": 0.12,
            "top_return_reasons": [
                {"reason": "not as described", "count": 456},
                {"reason": "defective", "count": 234},
                {"reason": "changed mind", "count": 189}
            ],
            "anomaly_detection_stats": {
                "total_analyzed": 1245,
                "anomalies_detected": 89,
                "false_positive_rate": 0.05,
                "average_confidence": 0.87
            },
            "generated_at": datetime.now().isoformat()
        }
        
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps(statistics, indent=2)
            )]
        )
    
    else:
        raise ValueError(f"Unknown tool: {request.name}")

async def main():
    """Main entry point"""
    # Initialize return anomaly agent
    await return_agent.initialize()
    
    # Start Kafka consumer in background
    kafka_task = asyncio.create_task(return_agent.run_kafka_consumer())
    
    # Start MCP server
    async with stdio_server() as (read_stream, write_stream):
        server_task = asyncio.create_task(
            server.run(
                read_stream,
                write_stream,
                server.create_initialization_options()
            )
        )
        
        # Wait for either task to complete
        await asyncio.gather(kafka_task, server_task)

if __name__ == "__main__":
    asyncio.run(main())