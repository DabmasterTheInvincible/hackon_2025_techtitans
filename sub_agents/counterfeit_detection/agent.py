#!/usr/bin/env python3
"""
Counterfeit Detection Sub-Agent for TSCC
Processes MCP packets for counterfeit detection using MCP protocol
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Any

import redis.asyncio as redis
from kafka import KafkaConsumer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import joblib

from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, CallToolResult, CallToolRequest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CounterfeitDetectionResult:
    counterfeit_score: float
    indicators: List[str]
    confidence: float
    model_version: str
    processing_time: float

class CounterfeitDetectionAgent:
    def __init__(self):
        self.agent_name = "counterfeit_detection"
        self.redis_client = None
        self.model = None
        self.seller_model = None
        self.meta_model = None
        self.tokenizer = None
        self.bert = None
        self.classifier = None
        self.brand_keywords = {
            "luxury": ["gucci", "prada", "louis vuitton", "chanel", "hermes", "rolex", "cartier"],
            "electronics": ["apple", "samsung", "sony", "nike", "adidas"],
            "common_counterfeits": ["ray-ban", "oakley", "beats", "airpods"]
        }
        self.load_models()

    def load_models(self):
        """Load or initialize counterfeit detection models"""
        """try:
            self.model = joblib.load('/app/models/counterfeit_model.pkl')
            self.vectorizer = joblib.load('/app/models/counterfeit_vectorizer.pkl')
            logger.info("Loaded existing counterfeit detection models")
        except FileNotFoundError:
            self.model = LogisticRegression(random_state=42)
            self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
            logger.info("Initialized new counterfeit detection models")"""
        """try:
            # Load tokenizer and BERT base
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.bert = BertModel.from_pretrained("bert-base-uncased")
            
            # Load classifier head
            self.classifier = nn.Sequential(
                nn.Linear(self.bert.config.hidden_size, 1),
                nn.Sigmoid()
            )

            # Load fine-tuned weights
            state_dict = torch.load("models/counterfeit_bert_model.pt", map_location=torch.device("cpu"))
            self.classifier.load_state_dict(state_dict["classifier"])
            self.bert.load_state_dict(state_dict["bert"])

            self.bert.eval()
            self.classifier.eval()

            logger.info("✅ Loaded fine-tuned BERT + classifier")
        except Exception as e:
            logger.error(f"❌ Failed to load fine-tuned model: {e}")"""
        """Load or initialize counterfeit detection models"""
        try:
            # Load tokenizer
            self.tokenizer = BertTokenizer.from_pretrained("tokenizer", local_files_only=True)
            
            # Recreate the model architecture
            class BERTClassifier(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.bert = BertModel.from_pretrained('bert_base', local_files_only=True)
                    self.drop = nn.Dropout(0.3)
                    self.out = nn.Linear(self.bert.config.hidden_size, 1)
                
                def forward(self, input_ids, attention_mask):
                    outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                    cls = self.drop(outputs.pooler_output)
                    return self.out(cls).squeeze(1)
            
            # Create and load model
            self.model = BERTClassifier()
            state_dict = torch.load("bert_base/bert_classifier_head.pt", map_location=torch.device("cpu"))
            self.model.load_state_dict(state_dict)
            self.model.eval()
            
            # EXPOSE BERT SEPARATELY for your inference code
            self.bert = self.model.bert
            self.text_head = self.model.out
            
            # Load other models
            self.seller_model = joblib.load("models/seller_history_model.pkl")
            self.meta_model = joblib.load("models/meta_classifier.pkl")
            
            logger.info("✅ Successfully loaded all models")
            
        except Exception as e:
            logger.error(f"❌ Failed to load models: {e}")

    async def initialize(self):
        """Initialize connections"""
        self.redis_client = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))

    def extract_text_features(self, mcp_packet: Dict[str, Any]) -> np.ndarray:
        """Extract text features using TF-IDF vectorization"""
        """raw_data = mcp_packet.get("raw_data", {})
        
        # Combine title and description for analysis
        title = raw_data.get("title", "")
        description = raw_data.get("description", "")
        brand = raw_data.get("brand", "")
        
        combined_text = f"{title} {description} {brand}".strip()
        
        if not combined_text:
            # Return zero vector if no text available
            if hasattr(self.vectorizer, 'transform'):
                return self.vectorizer.transform([""])
            else:
                return np.zeros((1, 5000))
        
        try:
            if hasattr(self.vectorizer, 'transform'):
                return self.vectorizer.transform([combined_text])
            else:
                # Initialize vectorizer with dummy data if not fitted
                self.vectorizer.fit([combined_text, "dummy text"])
                return self.vectorizer.transform([combined_text])
        except Exception as e:
            logger.warning(f"Error in text feature extraction: {e}")
            return np.zeros((1, 5000))"""
        """Tokenize and encode product text using BERT"""
        raw_data = mcp_packet.get("raw_data", {})
        title = raw_data.get("title", "")
        description = raw_data.get("description", "")
        brand = raw_data.get("brand", "")
        text = f"{title} {description} {brand}".strip()

        if not text:
            text = "[PAD]"
        
        return self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )

    def detect_counterfeit(self, mcp_packet: Dict[str, Any]) -> CounterfeitDetectionResult:
        """Main counterfeit detection logic"""
        """start_time = datetime.now()
        
        try:
            # Extract text features
            text_features = self.extract_text_features(mcp_packet)
            
            # Get counterfeit probability
            if hasattr(self.model, 'predict_proba'):
                try:
                    proba = self.model.predict_proba(text_features)[0]
                    counterfeit_score = float(proba[1]) if len(proba) > 1 else 0.0
                    confidence = float(np.max(proba))
                except:
                    # Fallback if model not trained
                    counterfeit_score = 0.0
                    confidence = 0.0
            else:
                counterfeit_score = 0.0
                confidence = 0.0
            
            # Detect rule-based indicators
            indicators = self._detect_indicators(mcp_packet)
            
            # Adjust score based on indicators
            if indicators:
                indicator_boost = min(0.3, len(indicators) * 0.1)
                counterfeit_score = min(1.0, counterfeit_score + indicator_boost)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return CounterfeitDetectionResult(
                counterfeit_score=counterfeit_score,
                indicators=indicators,
                confidence=confidence,
                model_version="1.0.0",
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error in counterfeit detection: {e}")
            return CounterfeitDetectionResult(
                counterfeit_score=0.0,
                indicators=["processing_error"],
                confidence=0.0,
                model_version="1.0.0",
                processing_time=(datetime.now() - start_time).total_seconds()
            )"""
        """start_time = datetime.now()
        try:
            inputs = self.extract_text_features(mcp_packet)
            with torch.no_grad():
                outputs = self.bert(**inputs)
                cls_embedding = outputs.pooler_output  # [CLS] token
                score = self.classifier(cls_embedding).squeeze().item()

            indicators = self._detect_indicators(mcp_packet)
            if indicators:
                score = min(1.0, score + min(0.3, len(indicators) * 0.1))
            
            processing_time = (datetime.now() - start_time).total_seconds()
            return CounterfeitDetectionResult(
                counterfeit_score=score,
                indicators=indicators,
                confidence=score,
                model_version="bert-v1",
                processing_time=processing_time
            )

        except Exception as e:
            logger.error(f"Error in BERT counterfeit detection: {e}")
            return CounterfeitDetectionResult(
                counterfeit_score=0.0,
                indicators=["processing_error"],
                confidence=0.0,
                model_version="bert-v1",
                processing_time=(datetime.now() - start_time).total_seconds()
            )"""
        """Use stacked ensemble to predict counterfeit risk"""
        start_time = datetime.now()
        try:
            # Extract seller features
            raw = mcp_packet.get("raw_data", {})
            context = mcp_packet.get("context", {})

            seller_feats = np.array([[
                float(context.get("seller_history", {}).get("avg_rating", 5.0)),
                float(context.get("seller_history", {}).get("account_age_days", 90)),
                int(context.get("seller_similar_listings", 0)),
                float(raw.get("price", 0))
            ]])

            seller_score = self.seller_model.predict_proba(seller_feats)[:, 1]  # shape: (1,)

            # Extract BERT text embedding
            title = raw.get("title", "")
            desc = raw.get("description", "")
            brand = raw.get("brand", "")
            combined = f"{title} {desc} {brand}"

            tokens = self.tokenizer(combined, return_tensors="pt", truncation=True, padding=True, max_length=128)
            with torch.no_grad():
                out = self.bert(**tokens)
                cls_embedding = out.pooler_output  # (1, 768)

            # Concatenate seller score and CLS embedding
            X_ensemble = np.hstack([cls_embedding.numpy(), seller_score.reshape(-1, 1)])  # (1, 769)

            # Meta-model prediction
            final_score = float(self.meta_model.predict_proba(X_ensemble)[0, 1])

            # Rule-based indicators
            indicators = self._detect_indicators(mcp_packet)
            if indicators:
                final_score = min(1.0, final_score + min(0.3, len(indicators) * 0.1))

            return CounterfeitDetectionResult(
                counterfeit_score=final_score,
                indicators=indicators,
                confidence=final_score,
                model_version="ensemble-v1",
                processing_time=(datetime.now() - start_time).total_seconds()
            )

        except Exception as e:
            logger.error(f"❌ Error in ensemble detection: {e}")
            return CounterfeitDetectionResult(
                counterfeit_score=0.0,
                indicators=["processing_error"],
                confidence=0.0,
                model_version="ensemble-v1",
                processing_time=(datetime.now() - start_time).total_seconds()
            )

    def _detect_indicators(self, mcp_packet: Dict[str, Any]) -> List[str]:
        """Detect specific counterfeit indicators using rule-based approach"""
        indicators = []
        raw_data = mcp_packet.get("raw_data", {})
        context = mcp_packet.get("context", {})
        
        title = raw_data.get("title", "").lower()
        description = raw_data.get("description", "").lower()
        brand = raw_data.get("brand", "").lower()
        price = float(raw_data.get("price", 0))
        
        # Missing authenticity guarantees
        auth_keywords = ["authentic", "genuine", "original", "authorized", "official"]
        if not any(keyword in description for keyword in auth_keywords):
            indicators.append("missing_authenticity_guarantees")
        
        # Suspicious brand pricing
        for category, brands in self.brand_keywords.items():
            if any(b in brand or b in title for b in brands):
                if category == "luxury" and price < 100:
                    indicators.append("suspicious_luxury_pricing")
                elif category == "electronics" and price < 50:
                    indicators.append("suspicious_electronics_pricing")
        
        # Poor grammar/spelling in description
        common_errors = ["orignal", "gurantee", "authetic", "geniune", "qualty"]
        if any(error in description for error in common_errors):
            indicators.append("poor_spelling_grammar")
        
        # Vague product descriptions
        if len(description.split()) < 10:
            indicators.append("vague_description")
        
        # Stock photo indicators
        if "stock photo" in description or "image for reference" in description:
            indicators.append("stock_photo_usage")
        
        # Seller history indicators
        seller_history = context.get("seller_history", {})
        if seller_history.get("seller_age_days", 0) < 30:
            indicators.append("new_seller_account")
        
        if seller_history.get("seller_rating", 5.0) < 3.0:
            indicators.append("low_seller_rating")
        
        # Shipping location mismatch
        shipping_info = raw_data.get("shipping_origin", "")
        if brand in self.brand_keywords["luxury"] and "china" in shipping_info.lower():
            indicators.append("suspicious_shipping_origin")
        
        # Multiple similar listings
        if context.get("seller_similar_listings", 0) > 10:
            indicators.append("bulk_similar_listings")
        
        return indicators

    async def process_mcp_packet(self, mcp_packet: Dict[str, Any]):
        """Process MCP packet and store results"""
        trace_id = mcp_packet.get("trace_id")
        
        # Perform counterfeit detection
        result = self.detect_counterfeit(mcp_packet)
        
        # Update MCP packet with results
        mcp_packet["sub_agent_results"][self.agent_name] = {
            "counterfeit_score": result.counterfeit_score,
            "indicators": result.indicators,
            "confidence": result.confidence,
            "model_version": result.model_version,
            "processing_time": result.processing_time,
            "processed_at": datetime.now().isoformat()
        }
        
        mcp_packet["risk_scores"][self.agent_name] = result.counterfeit_score
        
        # Store updated MCP packet in Redis
        await self.redis_client.set(
            f"mcp_results:{trace_id}",
            json.dumps(mcp_packet, default=str),
            ex=3600
        )
        
        logger.info(f"Processed counterfeit detection for trace_id: {trace_id}, score: {result.counterfeit_score:.3f}")

    async def run_kafka_consumer(self):
        """Run Kafka consumer for processing events"""
        consumer = KafkaConsumer(
            'counterfeit-detection-topic',
            #bootstrap_servers=['kafka:9092'],
            bootstrap_servers=['localhost:9092'],
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            group_id='counterfeit-detection-group',
            auto_offset_reset='latest'
        )
        
        logger.info("Started Kafka consumer for counterfeit detection")
        
        for message in consumer:
            try:
                data = message.value
                mcp_packet = data.get("mcp_packet")
                
                if mcp_packet:
                    await self.process_mcp_packet(mcp_packet)
                
            except Exception as e:
                logger.error(f"Error processing Kafka message: {e}")

# MCP Server Implementation
server = Server("counterfeit-detection-agent")

# Global agent instance
counterfeit_agent = CounterfeitDetectionAgent()

@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available counterfeit detection tools."""
    return [
        Tool(
            name="analyze_counterfeit_risk",
            description="Analyze counterfeit risk for marketplace products",
            inputSchema={
                "type": "object",
                "properties": {
                    "product_data": {
                        "type": "object",
                        "description": "Product information including title, description, brand, price"
                    },
                    "context": {
                        "type": "object",
                        "description": "Additional context including seller history"
                    }
                },
                "required": ["product_data"]
            }
        ),
        Tool(
            name="get_counterfeit_indicators",
            description="Get common counterfeit indicators and patterns",
            inputSchema={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Product category (luxury, electronics, fashion, etc.)",
                        "enum": ["luxury", "electronics", "fashion", "general"]
                    }
                }
            }
        ),
        Tool(
            name="update_counterfeit_model",
            description="Update counterfeit detection model with new training data",
            inputSchema={
                "type": "object",
                "properties": {
                    "text_data": {
                        "type": "array",
                        "description": "Product text data for training"
                    },
                    "labels": {
                        "type": "array",
                        "description": "Counterfeit labels (0=genuine, 1=counterfeit)"
                    }
                },
                "required": ["text_data", "labels"]
            }
        ),
        Tool(
            name="check_brand_authenticity",
            description="Check if product claims match known authentic brand patterns",
            inputSchema={
                "type": "object",
                "properties": {
                    "brand": {
                        "type": "string",
                        "description": "Brand name to verify"
                    },
                    "product_details": {
                        "type": "object",
                        "description": "Product details including price, description, etc."
                    }
                },
                "required": ["brand", "product_details"]
            }
        )
    ]

@server.call_tool()
async def call_tool(request: CallToolRequest) -> CallToolResult:
    """Handle tool calls for counterfeit detection."""
    
    if request.name == "analyze_counterfeit_risk":
        product_data = request.arguments.get("product_data", {})
        context = request.arguments.get("context", {})
        
        # Create mock MCP packet for analysis
        mcp_packet = {
            "raw_data": product_data,
            "context": context,
            "timestamp": datetime.now().isoformat(),
            "sub_agent_results": {},
            "risk_scores": {}
        }
        
        result = counterfeit_agent.detect_counterfeit(mcp_packet)
        
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps({
                    "counterfeit_score": result.counterfeit_score,
                    "indicators": result.indicators,
                    "confidence": result.confidence,
                    "model_version": result.model_version,
                    "risk_level": "high" if result.counterfeit_score > 0.7 else "medium" if result.counterfeit_score > 0.3 else "low"
                }, indent=2)
            )]
        )
    
    elif request.name == "get_counterfeit_indicators":
        category = request.arguments.get("category", "general")
        
        indicators = {
            "luxury": [
                "Significantly below market price",
                "Poor quality images or stock photos",
                "Vague product descriptions",
                "Missing authenticity certificates",
                "Shipping from unexpected locations",
                "New seller accounts",
                "Bulk similar listings"
            ],
            "electronics": [
                "Prices too good to be true",
                "Missing serial numbers or warranty info",
                "Poor packaging descriptions",
                "Non-authorized retailers",
                "Spelling errors in brand names",
                "Generic product images"
            ],
            "fashion": [
                "Incorrect logos or branding",
                "Poor material descriptions",
                "Missing size charts or fit info",
                "Unusually low prices for designer items",
                "Stock photo usage",
                "Vague origin information"
            ],
            "general": [
                "Missing authenticity guarantees",
                "Poor spelling and grammar",
                "Vague product descriptions",
                "Suspicious pricing",
                "New seller accounts",
                "Low seller ratings",
                "Stock photo indicators"
            ]
        }
        
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps({
                    "category": category,
                    "indicators": indicators.get(category, indicators["general"]),
                    "detection_tips": [
                        "Compare prices with authorized retailers",
                        "Check seller history and ratings",
                        "Look for detailed product descriptions",
                        "Verify authenticity guarantees",
                        "Check shipping origins"
                    ]
                }, indent=2)
            )]
        )
    elif request.name == "update_counterfeit_model":
        text_data = request.arguments.get("text_data", [])
        seller_features = request.arguments.get("seller_features", [])
        labels = request.arguments.get("labels", [])

        if not (len(text_data) == len(labels) == len(seller_features)):
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text="Error: text_data, seller_features, and labels must have the same length."
                )]
            )

        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            import numpy as np
            from sklearn.linear_model import LogisticRegression
            from sklearn.ensemble import GradientBoostingClassifier
            import joblib
            import os

            tokenizer = counterfeit_agent.tokenizer
            bert = counterfeit_agent.bert
            bert.eval()

            for param in bert.parameters():
                param.requires_grad = False

            # Define classifier head (same as used in loading)
            classifier_head = nn.Linear(bert.config.hidden_size, 1)
            loss_fn = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(classifier_head.parameters(), lr=1e-4)

            # Tokenize and encode
            encoded_inputs = [tokenizer(t, return_tensors="pt", padding="max_length", truncation=True, max_length=128) for t in text_data]
            labels_tensor = torch.tensor(labels).float()

            classifier_head.train()
            for epoch in range(3):
                total_loss = 0.0
                for i in range(len(encoded_inputs)):
                    inp = encoded_inputs[i]
                    with torch.no_grad():
                        out = bert(**{k: v for k, v in inp.items()})
                    cls_embedding = out.pooler_output
                    logit = classifier_head(cls_embedding).squeeze()
                    loss = loss_fn(logit, labels_tensor[i])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

            # Extract embeddings
            classifier_head.eval()
            text_embeddings = []
            with torch.no_grad():
                for inp in encoded_inputs:
                    out = bert(**{k: v for k, v in inp.items()})
                    cls_embedding = out.pooler_output.squeeze().numpy()
                    text_embeddings.append(cls_embedding)

            text_embeddings = np.array(text_embeddings)

            # Train seller model
            seller_features_array = np.array(seller_features)
            seller_model = GradientBoostingClassifier()
            seller_model.fit(seller_features_array, labels)

            seller_probs = seller_model.predict_proba(seller_features_array)[:, 1].reshape(-1, 1)

            # Meta-classifier
            final_inputs = np.hstack([text_embeddings, seller_probs])
            meta_model = LogisticRegression(max_iter=200)
            meta_model.fit(final_inputs, labels)

            # Save everything
            os.makedirs("bert_base", exist_ok=True)
            os.makedirs("models", exist_ok=True)
            torch.save(classifier_head.state_dict(), "bert_base/bert_classifier_head.pt")
            joblib.dump(seller_model, "models/seller_model.pkl")
            joblib.dump(meta_model, "models/meta_classifier.pkl")

            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps({
                        "status": "success",
                        "message": f"Ensemble updated with {len(labels)} samples",
                        "timestamp": datetime.now().isoformat()
                    }, indent=2)
                )]
            )

        except Exception as e:
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=f"Error during retraining: {str(e)}"
                )]
            )
    elif request.name == "check_brand_authenticity":
        brand = request.arguments.get("brand", "").lower()
        product_details = request.arguments.get("product_details", {})
        
        authenticity_check = {
            "brand": brand,
            "is_known_brand": False,
            "category": "unknown",
            "risk_factors": [],
            "authenticity_score": 0.5
        }
        
        # Check against known brands
        for category, brands in counterfeit_agent.brand_keywords.items():
            if brand in brands:
                authenticity_check["is_known_brand"] = True
                authenticity_check["category"] = category
                break
        
        if authenticity_check["is_known_brand"]:
            price = float(product_details.get("price", 0))
            
            # Check pricing consistency
            if authenticity_check["category"] == "luxury" and price < 100:
                authenticity_check["risk_factors"].append("price_too_low_for_luxury")
                authenticity_check["authenticity_score"] = 0.2
            elif authenticity_check["category"] == "electronics" and price < 50:
                authenticity_check["risk_factors"].append("price_too_low_for_electronics")
                authenticity_check["authenticity_score"] = 0.3
            else:
                authenticity_check["authenticity_score"] = 0.8
            
            # Check description quality
            description = product_details.get("description", "")
            if len(description.split()) < 10:
                authenticity_check["risk_factors"].append("insufficient_description")
                authenticity_check["authenticity_score"] *= 0.8
        
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps(authenticity_check, indent=2)
            )]
        )
    
    else:
        raise ValueError(f"Unknown tool: {request.name}")

async def main():
    """Main entry point"""
    # Initialize counterfeit agent
    await counterfeit_agent.initialize()
    
    # Start Kafka consumer in background
    kafka_task = asyncio.create_task(counterfeit_agent.run_kafka_consumer())
    
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