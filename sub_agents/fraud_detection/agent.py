#!/usr/bin/env python3
"""
Fraud Detection Sub-Agent for TSCC
Processes MCP packets for fraud detection using MCP protocol
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, List
from dataclasses import dataclass
import os

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, CallToolResult, CallToolRequest

import redis.asyncio as redis
from kafka import KafkaConsumer
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import torch
import torch
import torch.nn as nn
import torch.optim as optim
import joblib

class Autoencoder(nn.Module):
        def __init__(self, input_dim):
            super(Autoencoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 16),
                nn.ReLU(),
                nn.Linear(16, 3)
            )
            self.decoder = nn.Sequential(
                nn.Linear(3, 16),
                nn.ReLU(),
                nn.Linear(16, 64),
                nn.ReLU(),
                nn.Linear(64, input_dim),
                nn.Sigmoid()
            )

        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x
        
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FraudDetectionResult:
    fraud_score: float
    risk_indicators: List[str]
    confidence: float
    model_version: str
    processing_time: float

class FraudDetectionAgent:
    def __init__(self):
        self.agent_name = "fraud_detection"
        self.redis_client = None
        self.model = None
        self.scaler = None
        self.min_score = 0.051617
        self.max_score = 0.125103
        self.load_models()
    
    def load_models(self):
        try:
            # Load scaler
            self.scaler = joblib.load("models/scaler.pkl")

            # Load model architecture
            input_dim = self.scaler.n_features_in_
            self.model = Autoencoder(input_dim)
            self.model.load_state_dict(torch.load("models/autoencoder.pth", map_location=torch.device('cpu')))
            self.model.eval()

            logger.info("Autoencoder and scaler loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

    async def initialize(self):
        """Initialize connections"""
        self.redis_client = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))

    def extract_features(self, mcp_packet: Dict[str, Any]) -> np.ndarray:
        """Extract features from MCP packet for fraud detection"""
        raw_data = mcp_packet.get("raw_data", {})
        context = mcp_packet.get("context", {})
        
        features = []
        
        # Basic event features
        features.append(len(str(raw_data.get("product_id", ""))))
        features.append(len(str(raw_data.get("description", ""))))
        features.append(float(raw_data.get("price", 0)))
        features.append(float(raw_data.get("quantity", 1)))
        
        # Seller features
        seller_history = context.get("seller_history", {})
        features.append(float(seller_history.get("total_sales", 0)))
        features.append(float(seller_history.get("avg_rating", 0)))
        features.append(float(seller_history.get("total_reviews", 0)))
        features.append(float(seller_history.get("account_age_days", 0)))
        
        # Customer features
        customer_history = context.get("customer_history", {})
        features.append(float(customer_history.get("total_purchases", 0)))
        features.append(float(customer_history.get("avg_order_value", 0)))
        features.append(float(customer_history.get("return_rate", 0)))
        
        # Time-based features
        timestamp = mcp_packet.get("timestamp", datetime.now().isoformat())
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            features.append(float(dt.hour))
            features.append(float(dt.weekday()))
        except:
            features.extend([0.0, 0.0])
        
        # Behavioral patterns
        features.append(float(raw_data.get("is_expedited_shipping", False)))
        features.append(float(raw_data.get("payment_method_new", False)))
        features.append(float(raw_data.get("shipping_address_new", False)))
        
        return np.array(features).reshape(1, -1)

    def detect_fraud(self, mcp_packet: Dict[str, Any]) -> FraudDetectionResult:
        start_time = datetime.now()

        try:
            features = self.extract_features(mcp_packet)
            features_scaled = self.scaler.transform(features)

            # Convert to tensor and infer
            input_tensor = torch.tensor(features_scaled, dtype=torch.float32)
            with torch.no_grad():
                reconstructed = self.model(input_tensor).numpy()

            # MSE loss between input and reconstruction
            mse = np.mean(np.square(features_scaled - reconstructed), axis=1)[0]
            logger.info(f"Reconstruction MSE: {mse:.6f}")

            # Normalize fraud score between 0–1
            fraud_score = (mse - self.min_score) / (self.max_score - self.min_score)
            fraud_score = max(0.0, min(1.0, fraud_score))

            risk_indicators = self._identify_risk_indicators(mcp_packet, features)
            confidence = self._calculate_confidence(features)
            processing_time = (datetime.now() - start_time).total_seconds()

            return FraudDetectionResult(
                fraud_score=fraud_score,
                risk_indicators=risk_indicators,
                confidence=confidence,
                model_version="autoencoder-pt-1.0",
                processing_time=processing_time
            )

        except Exception as e:
            logger.error(f"Error in fraud detection: {e}")
            return FraudDetectionResult(
                fraud_score=0.0,
                risk_indicators=["processing_error"],
                confidence=0.0,
                model_version="autoencoder-pt-1.0",
                processing_time=(datetime.now() - start_time).total_seconds()
            )


    def _identify_risk_indicators(self, mcp_packet: Dict[str, Any], features: np.ndarray) -> List[str]:
        """Identify specific risk indicators"""
        indicators = []
        raw_data = mcp_packet.get("raw_data", {})
        context = mcp_packet.get("context", {})
        
        # High-value transaction
        if raw_data.get("price", 0) > 1000:
            indicators.append("high_value_transaction")
        
        # New seller
        seller_history = context.get("seller_history", {})
        if seller_history.get("account_age_days", 0) < 30:
            indicators.append("new_seller_account")
        
        # Unusual shipping
        if raw_data.get("is_expedited_shipping") and raw_data.get("price", 0) < 50:
            indicators.append("unusual_shipping_pattern")
        
        # Multiple new factors
        new_factors = sum([
            raw_data.get("payment_method_new", False),
            raw_data.get("shipping_address_new", False),
            seller_history.get("account_age_days", 0) < 30
        ])
        if new_factors >= 2:
            indicators.append("multiple_new_factors")
        
        # Off-hours activity
        try:
            timestamp = mcp_packet.get("timestamp", datetime.now().isoformat())
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            if dt.hour < 6 or dt.hour > 22:
                indicators.append("off_hours_activity")
        except:
            pass
        
        return indicators

    def _calculate_confidence(self, features: np.ndarray) -> float:
        """Calculate confidence based on feature completeness"""
        # Count non-zero features as a proxy for data completeness
        non_zero_features = np.count_nonzero(features)
        total_features = features.shape[1]
        
        # Base confidence on data completeness
        confidence = non_zero_features / total_features
        
        # Adjust for known important features
        if features[0, 2] > 0:  # Price available
            confidence += 0.1
        if features[0, 4] > 0:  # Seller history available
            confidence += 0.1
        if features[0, 8] > 0:  # Customer history available
            confidence += 0.1
        
        return min(1.0, confidence)

    async def process_mcp_packet(self, mcp_packet: Dict[str, Any]):
        """Process MCP packet and store results"""
        trace_id = mcp_packet.get("trace_id")
        
        # Perform fraud detection
        result = self.detect_fraud(mcp_packet)
        
        # Update MCP packet with results
        mcp_packet["sub_agent_results"][self.agent_name] = {
            "fraud_score": result.fraud_score,
            "risk_indicators": result.risk_indicators,
            "confidence": result.confidence,
            "model_version": result.model_version,
            "processing_time": result.processing_time,
            "processed_at": datetime.now().isoformat()
        }
        
        mcp_packet["risk_scores"][self.agent_name] = result.fraud_score
        
        # Store updated MCP packet in Redis
        await self.redis_client.set(
            f"mcp_results:{trace_id}",
            json.dumps(mcp_packet, default=str),
            ex=3600
        )
        
        logger.info(f"Processed fraud detection for trace_id: {trace_id}, score: {result.fraud_score:.3f}")

    async def run_kafka_consumer(self):
        """Run Kafka consumer for processing events"""
        consumer = KafkaConsumer(
            'fraud-detection-topic',
            #bootstrap_servers=['kafka:9092'],
            bootstrap_servers=['localhost:9092'],
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            group_id='fraud-detection-group',
            auto_offset_reset='latest'
        )
        
        logger.info("Started Kafka consumer for fraud detection")
        
        for message in consumer:
            try:
                data = message.value
                mcp_packet = data.get("mcp_packet")
                
                if mcp_packet:
                    await self.process_mcp_packet(mcp_packet)
                
            except Exception as e:
                logger.error(f"Error processing Kafka message: {e}")

# MCP Server Implementation
server = Server("fraud-detection-agent")

# Global agent instance
fraud_agent = FraudDetectionAgent()

@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available fraud detection tools."""
    return [
        Tool(
            name="analyze_fraud_risk",
            description="Analyze fraud risk for marketplace events",
            inputSchema={
                "type": "object",
                "properties": {
                    "event_data": {
                        "type": "object",
                        "description": "Marketplace event data"
                    },
                    "context": {
                        "type": "object", 
                        "description": "Additional context data"
                    }
                },
                "required": ["event_data"]
            }
        ),
        Tool(
            name="get_fraud_patterns",
            description="Get common fraud patterns and indicators",
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern_type": {
                        "type": "string",
                        "description": "Type of fraud pattern to retrieve"
                    }
                }
            }
        ),
        Tool(
            name="update_fraud_model",
            description="Update fraud detection model with new training data",
            inputSchema={
                "type": "object",
                "properties": {
                    "training_data": {
                        "type": "array",
                        "description": "New training data for model update"
                    },
                    "labels": {
                        "type": "array",
                        "description": "Labels for training data"
                    }
                },
                "required": ["training_data", "labels"]
            }
        )
    ]

@server.call_tool()
async def call_tool(request: CallToolRequest) -> CallToolResult:
    """Handle tool calls for fraud detection."""
    
    if request.name == "analyze_fraud_risk":
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
        
        result = fraud_agent.detect_fraud(mcp_packet)
        
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps({
                    "fraud_score": result.fraud_score,
                    "risk_indicators": result.risk_indicators,
                    "confidence": result.confidence,
                    "model_version": result.model_version
                }, indent=2)
            )]
        )
    
    elif request.name == "get_fraud_patterns":
        pattern_type = request.arguments.get("pattern_type", "common")
        
        patterns = {
            "common": [
                "High-value transactions from new accounts",
                "Multiple payment method changes",
                "Expedited shipping on low-value items",
                "Off-hours transaction patterns",
                "Rapid succession purchases"
            ],
            "seller": [
                "New seller with premium products",
                "Sellers with no review history",
                "Price significantly below market rate",
                "Duplicate product listings"
            ],
            "buyer": [
                "New shipping addresses frequently",
                "High return rates",
                "Unusual payment patterns",
                "Multiple failed payment attempts"
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
    
    elif request.name == "update_fraud_model":
        training_data = request.arguments.get("training_data", [])

        if not training_data:
            return CallToolResult(
                content=[TextContent(type="text", text="Error: No training data provided.")]
            )

        try:
            import numpy as np
            from sklearn.preprocessing import MinMaxScaler
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import TensorDataset, DataLoader
            import os

            X = np.array(training_data, dtype=np.float32)

            # Scale features
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)

            # PyTorch Dataset
            tensor_data = torch.tensor(X_scaled, dtype=torch.float32)
            dataset = TensorDataset(tensor_data, tensor_data)
            loader = DataLoader(dataset, batch_size=64, shuffle=True)

            # Model and training
            input_dim = X.shape[1]
            model = Autoencoder(input_dim)
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.MSELoss()

            for epoch in range(20):
                for batch in loader:
                    inputs, targets = batch
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # Save model and scaler
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/autoencoder.pth")
            joblib.dump(scaler, "models/scaler.pkl")

            return {
                "status": "success",
                "message": f"Model updated with {len(training_data)} samples",
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    
    else:
        raise ValueError(f"Unknown tool: {request.name}")

async def main():
    """Main entry point"""
    # Initialize fraud agent
    await fraud_agent.initialize()
    
    # Start Kafka consumer in background
    kafka_task = asyncio.create_task(fraud_agent.run_kafka_consumer())
    
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