#!/usr/bin/env python3
"""
Super Agent for Trust & Safety Command Center (TSCC)
Main FastAPI application entry point
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from mcp_bridge import run_mcp

from mcp_protocol import MCPPacket, TSCCSuperAgent, RiskLevel
from event_router import MarketplaceEvent
from prometheus_client import start_http_server, Summary, Counter, Gauge

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="TSCC Super Agent", version="1.0.0")

EVENT_PROCESSING_TIME = Summary('event_processing_time_seconds', 'Time taken to process an event')
DETECTIONS_TOTAL = Counter('detections_total', 'Total detections by risk level', ['risk_level'])
FALSE_POSITIVE_TOTAL = Counter('false_positives_total', 'False positive reports per agent', ['agent'])
ACTIVE_CASES = Gauge('active_cases', 'Number of cases under investigation')

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalystAction(BaseModel):
    trace_id: str
    action: str   # "confirmed_fraud", "false_positive", "escalate"
    agent: str    # e.g., "fraud_detection"

# Initialize global agent
super_agent = TSCCSuperAgent()

@app.post("/run-mcp")
async def handle_mcp(request: Request):
    packet = await request.json()
    result = run_mcp(packet)
    return result

@app.on_event("startup")
async def startup_event():
    """Initialize connections on startup"""
    await super_agent.initialize()
    logger.info("TSCC Super Agent initialized successfully")
    start_http_server(9000)

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up connections on shutdown"""
    await super_agent.cleanup()
    logger.info("TSCC Super Agent shutdown complete")

@app.post("/analyst_action")
async def analyst_action(action: AnalystAction):
    if action.action == "false_positive":
        FALSE_POSITIVE_TOTAL.labels(agent=action.agent).inc()
        ACTIVE_CASES.dec()

    elif action.action in ["confirmed_fraud", "escalate"]:
        ACTIVE_CASES.dec()

    return {"status": "acknowledged"}

@app.post("/process_event")
async def process_marketplace_event(event: MarketplaceEvent, background_tasks: BackgroundTasks):
    """Main endpoint to process marketplace events"""
    try:
        # Create MCP packet
        mcp_packet = await super_agent.create_mcp_packet(event)
        
        # Route to sub-agents
        await super_agent.route_to_sub_agents(mcp_packet)
        
        # Store initial packet in Redis
        await super_agent.redis_client.set(
            f"mcp_results:{mcp_packet.trace_id}",
            json.dumps(mcp_packet.to_dict(), default=str),
            ex=3600  # 1 hour expiry
        )
        
        # Schedule result aggregation
        background_tasks.add_task(process_results, mcp_packet.trace_id)
        
        logger.info(f"Started processing event {mcp_packet.event_id} with trace_id {mcp_packet.trace_id}")
        
        return {
            "status": "processing",
            "trace_id": mcp_packet.trace_id,
            "event_id": mcp_packet.event_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error processing event: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process event: {str(e)}")

async def process_results(trace_id: str):
    """Background task to process aggregated results"""
    try:
        with EVENT_PROCESSING_TIME.time():
            logger.info(f"Starting result aggregation for trace_id: {trace_id}")
            
            # Wait for sub-agent results
            mcp_packet = await super_agent.aggregate_results(trace_id)
            
            # Calculate final risk
            mcp_packet.final_risk_level = await super_agent.calculate_final_risk(mcp_packet)
            
            # Generate investigation brief if needed
            mcp_packet.investigation_brief = await super_agent.generate_investigation_brief(mcp_packet)
            
            # Store final result
            await super_agent.redis_client.set(
                f"mcp_final:{trace_id}",
                json.dumps(mcp_packet.to_dict(), default=str),
                ex=86400  # 24 hours expiry
            )
            
            # If high risk, publish to high-priority topic
            if mcp_packet.final_risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                await super_agent.publish_high_priority_alert(mcp_packet)
            
            logger.info(f"Completed processing for trace_id: {trace_id} with risk level: {mcp_packet.final_risk_level.value}")
            DETECTIONS_TOTAL.labels(risk_level=mcp_packet.final_risk_level.value).inc()
            if mcp_packet.final_risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                ACTIVE_CASES.inc()
    except Exception as e:
        logger.error(f"Error in background processing for trace_id {trace_id}: {e}")

@app.get("/results/{trace_id}")
async def get_results(trace_id: str):
    """Get processing results for a trace ID"""
    try:
        final_key = f"mcp_final:{trace_id}"
        result = await super_agent.redis_client.get(final_key)
        
        if result:
            return json.loads(result)
        
        # Check if still processing
        processing_key = f"mcp_results:{trace_id}"
        processing_result = await super_agent.redis_client.get(processing_key)
        
        if processing_result:
            return {
                "status": "processing", 
                "partial_results": json.loads(processing_result),
                "message": "Results are still being processed"
            }
        
        raise HTTPException(status_code=404, detail=f"No results found for trace_id: {trace_id}")
        
    except Exception as e:
        logger.error(f"Error retrieving results for trace_id {trace_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve results")

@app.get("/status/{trace_id}")
async def get_processing_status(trace_id: str):
    """Get current processing status for a trace ID"""
    try:
        # Check for final results first
        final_key = f"mcp_final:{trace_id}"
        final_result = await super_agent.redis_client.get(final_key)
        
        if final_result:
            result_data = json.loads(final_result)
            return {
                "status": "completed",
                "risk_level": result_data.get("final_risk_level"),
                "completed_at": result_data.get("timestamp")
            }
        
        # Check for processing results
        processing_key = f"mcp_results:{trace_id}"
        processing_result = await super_agent.redis_client.get(processing_key)
        
        if processing_result:
            result_data = json.loads(processing_result)
            completed_agents = list(result_data.get("sub_agent_results", {}).keys())
            
            return {
                "status": "processing",
                "completed_agents": completed_agents,
                "started_at": result_data.get("timestamp")
            }
        
        return {"status": "not_found", "message": f"No processing found for trace_id: {trace_id}"}
        
    except Exception as e:
        logger.error(f"Error getting status for trace_id {trace_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get processing status")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test Redis connection
        redis_status = "healthy" if await super_agent.test_redis_connection() else "unhealthy"
        
        # Test Kafka connection
        kafka_status = "healthy" if super_agent.test_kafka_connection() else "unhealthy"
        
        overall_status = "healthy" if redis_status == "healthy" and kafka_status == "healthy" else "degraded"
        
        return {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "components": {
                "redis": redis_status,
                "kafka": kafka_status
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@app.get("/metrics")
async def get_metrics():
    """Get system metrics"""
    try:
        metrics = await super_agent.get_metrics()
        return metrics
    except Exception as e:
        logger.error(f"Error retrieving metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "TSCC Super Agent",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/active_traces")
async def get_active_trace_ids():
    try:
        keys = await super_agent.redis_client.keys("mcp_final:*")
        trace_ids = [key.decode().split(":")[1] for key in keys]
        return {"trace_ids": trace_ids}
    except Exception as e:
        logger.error(f"Error retrieving active trace IDs: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch active trace IDs")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info",
        reload=False
    )