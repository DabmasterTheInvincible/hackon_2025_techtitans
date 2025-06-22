#!/usr/bin/env python3
"""
Review Spam Detection Sub-Agent for TSCC
Processes MCP packets for review spam detection using MCP protocol
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
from textblob import TextBlob
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ReviewSpamResult:
    spam_score: float
    spam_signals: List[str]
    confidence: float
    model_version: str
    processing_time: float

class ReviewSpamAgent:
    def __init__(self):
        self.agent_name = "review_spam"
        self.redis_client = None
        self.known_spam_patterns = self._load_spam_patterns()
        self.tokenizer = None
        self.roberta_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def _load_spam_patterns(self) -> List[str]:
        """Load known spam patterns"""
        return [
            r"(best|worst|amazing|terrible)\s+product\s+ever",
            r"(buy|purchase|order)\s+now",
            r"(highly|strongly)\s+recommend",
            r"changed\s+my\s+life",
            r"money\s+back\s+guarantee",
            r"limited\s+time\s+offer",
            r"click\s+here",
            r"free\s+shipping"
        ]

    async def initialize(self):
        """Initialize connections"""
        self.redis_client = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
        self.tokenizer = RobertaTokenizer.from_pretrained("/models/roberta_spam_model/")
        self.roberta_model = RobertaForSequenceClassification.from_pretrained("/models/roberta_spam_model/")
        self.roberta_model.to(self.device)
        self.roberta_model.eval()

    def ml_model_spam_score(self, text: str) -> float:
        """Use fine-tuned RoBERTa model to compute spam probability"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.roberta_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            spam_prob = probs[0][1].item()
        
        return spam_prob

    def analyze_text_patterns(self, text: str) -> List[str]:
        """Analyze text for spam patterns"""
        signals = []
        text_lower = text.lower()
        
        # Check for known spam patterns
        for pattern in self.known_spam_patterns:
            if re.search(pattern, text_lower):
                signals.append("spam_pattern_detected")
                break
        
        # Check for excessive punctuation
        if len(re.findall(r'[!?]{2,}', text)) > 0:
            signals.append("excessive_punctuation")
        
        # Check for all caps
        if len([word for word in text.split() if word.isupper() and len(word) > 2]) > 2:
            signals.append("excessive_caps")
        
        # Check for repeated characters
        if re.search(r'(.)\1{3,}', text):
            signals.append("repeated_characters")
        
        # Check for promotional language
        promo_words = ["discount", "sale", "deal", "offer", "bargain", "cheap", "free"]
        if sum(1 for word in promo_words if word in text_lower) >= 2:
            signals.append("promotional_language")
        
        return signals

    def analyze_review_authenticity(self, mcp_packet: Dict[str, Any]) -> List[str]:
        """Analyze review for authenticity signals"""
        signals = []
        raw_data = mcp_packet.get("raw_data", {})
        context = mcp_packet.get("context", {})
        
        review_text = raw_data.get("review_text", "")
        rating = raw_data.get("rating", 0)
        user_id = raw_data.get("user_id", "")
        
        # Check review length vs rating correlation
        if rating >= 4 and len(review_text.strip()) < 20:
            signals.append("short_positive_review")
        elif rating <= 2 and len(review_text.strip()) < 30:
            signals.append("short_negative_review")
        
        # Check for fake urgency
        urgency_words = ["urgent", "hurry", "limited", "expires", "deadline"]
        if any(word in review_text.lower() for word in urgency_words):
            signals.append("fake_urgency")
        
        # Check user history
        user_history = context.get("customer_history", {})
        review_count = user_history.get("total_reviews", 0)
        account_age = user_history.get("account_age_days", 0)
        
        if review_count > 20 and account_age < 30:
            signals.append("suspicious_review_frequency")
        
        # Check for review bombing patterns
        if review_count > 10:
            past_reviews = user_history.get("past_reviews", [])
            recent_reviews = [r for r in past_reviews if r.get("days_ago", 999) < 7]
            if len(recent_reviews) > 5:
                signals.append("review_bombing_pattern")
        
        return signals

    def detect_spam(self, mcp_packet: Dict[str, Any]) -> ReviewSpamResult:
        """Main spam detection logic"""
        start_time = datetime.now()
        
        try:
            raw_data = mcp_packet.get("raw_data", {})
            review_text = raw_data.get("review_text", "")
            # ML-based score
            user_id = raw_data.get("user_id", "")
            context = mcp_packet.get("context", {})
            
            spam_signals = []
            spam_score = 0.0
            confidence = 0.8  # default confidence
            
            # Signal 1: Very short review
            if len(review_text.strip()) < 10:
                spam_signals.append("too_short")
                spam_score += 0.3
            
            # Signal 2: Overly generic text
            generic_reviews = ["good", "nice", "ok", "great", "bad", "fine", "awesome", "terrible"]
            if review_text.lower().strip() in generic_reviews:
                spam_signals.append("generic_review")
                spam_score += 0.3
            
            # Signal 3: Repeated content
            user_history = context.get("customer_history", {})
            past_reviews = user_history.get("past_reviews", [])
            past_review_texts = [r.get("text", "") for r in past_reviews if isinstance(r, dict)]
            
            if review_text in past_review_texts:
                spam_signals.append("duplicate_review")
                spam_score += 0.4
            
            # Signal 4: Sentiment analysis
            try:
                sentiment = TextBlob(review_text).sentiment
                polarity = sentiment.polarity
                subjectivity = sentiment.subjectivity
                
                # Neutral sentiment in detailed review might be suspicious
                if abs(polarity) < 0.1 and len(review_text) > 50:
                    spam_signals.append("neutral_sentiment_long_review")
                    spam_score += 0.2
                
                # Very subjective language might indicate fake review
                if subjectivity > 0.8:
                    spam_signals.append("highly_subjective")
                    spam_score += 0.15 
                    
            except Exception as e:
                logger.warning(f"Sentiment analysis failed: {e}")
                spam_signals.append("sentiment_analysis_failed")
                confidence -= 0.1
            
            # Signal 5: Text pattern analysis
            pattern_signals = self.analyze_text_patterns(review_text)
            spam_signals.extend(pattern_signals)
            spam_score += len(pattern_signals) * 0.1
            
            # Signal 6: Authenticity analysis
            auth_signals = self.analyze_review_authenticity(mcp_packet)
            spam_signals.extend(auth_signals)
            spam_score += len(auth_signals) * 0.15
            
            # Signal 7: Language quality analysis
            if len(review_text) > 20:
                # Check for grammatical issues (simple heuristic)
                sentences = review_text.split('.')
                avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
                
                if avg_sentence_length < 3:  # Very short sentences might indicate poor quality
                    spam_signals.append("poor_sentence_structure")
                    spam_score += 0.1
            
            # Signal 8: Timing analysis
            try:
                timestamp = mcp_packet.get("timestamp", datetime.now().isoformat())
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                
                # Reviews posted at unusual hours
                if dt.hour < 5 or dt.hour > 23:
                    spam_signals.append("unusual_posting_time")
                    spam_score += 0.05
                    
            except Exception:
                pass
            
            ml_spam_score = self.ml_model_spam_score(review_text)
            spam_score += ml_spam_score * 0.6  # 60% weight to ML model
            spam_signals.append("ml_model_score: {:.3f}".format(ml_spam_score))
            # Normalize spam score
            spam_score = min(spam_score, 1.0)
            
            # Adjust confidence based on available data
            if not review_text:
                confidence = 0.1
            elif len(user_history.get("past_reviews", [])) == 0:
                confidence -= 0.2
            
            confidence = max(0.1, min(1.0, confidence))
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ReviewSpamResult(
                spam_score=spam_score,
                spam_signals=spam_signals,
                confidence=confidence,
                model_version="1.0.0",
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error in spam detection: {e}")
            return ReviewSpamResult(
                spam_score=0.0,
                spam_signals=["processing_error"],
                confidence=0.0,
                model_version="1.0.0",
                processing_time=(datetime.now() - start_time).total_seconds()
            )

    async def process_mcp_packet(self, mcp_packet: Dict[str, Any]):
        """Process MCP packet and store results"""
        trace_id = mcp_packet.get("trace_id")
        
        # Perform spam detection
        result = self.detect_spam(mcp_packet)
        
        # Update MCP packet with results
        mcp_packet["sub_agent_results"][self.agent_name] = {
            "spam_score": result.spam_score,
            "spam_signals": result.spam_signals,
            "confidence": result.confidence,
            "model_version": result.model_version,
            "processing_time": result.processing_time,
            "processed_at": datetime.now().isoformat()
        }
        
        mcp_packet["risk_scores"][self.agent_name] = result.spam_score
        
        # Store updated MCP packet in Redis
        await self.redis_client.set(
            f"mcp_results:{trace_id}",
            json.dumps(mcp_packet, default=str),
            ex=3600
        )
        
        logger.info(f"Processed review spam for trace_id: {trace_id}, score: {result.spam_score:.3f}")

    async def run_kafka_consumer(self):
        """Run Kafka consumer for processing events"""
        consumer = KafkaConsumer(
            'review-spam-topic',
            #bootstrap_servers=['kafka:9092'],
            bootstrap_servers=['localhost:9092'],
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            group_id='review-spam-group',
            auto_offset_reset='latest'
        )
        
        logger.info("Started Kafka consumer for review spam detection")
        
        for message in consumer:
            try:
                data = message.value
                mcp_packet = data.get("mcp_packet")
                
                if mcp_packet:
                    await self.process_mcp_packet(mcp_packet)
                
            except Exception as e:
                logger.error(f"Error processing Kafka message: {e}")

# MCP Server Implementation
server = Server("review-spam-agent")

# Global agent instance
spam_agent = ReviewSpamAgent()

@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available review spam detection tools."""
    return [
        Tool(
            name="analyze_review_spam",
            description="Analyze review text for spam indicators",
            inputSchema={
                "type": "object",
                "properties": {
                    "review_data": {
                        "type": "object",
                        "description": "Review data including text, rating, user info",
                        "properties": {
                            "review_text": {"type": "string"},
                            "rating": {"type": "number"},
                            "user_id": {"type": "string"}
                        },
                        "required": ["review_text"]
                    },
                    "context": {
                        "type": "object",
                        "description": "Additional context data including user history"
                    }
                },
                "required": ["review_data"]
            }
        ),
        Tool(
            name="get_spam_patterns",
            description="Get known spam patterns and indicators",
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern_category": {
                        "type": "string",
                        "enum": ["text", "behavioral", "temporal", "all"],
                        "description": "Category of spam patterns to retrieve"
                    }
                }
            }
        ),
        Tool(
            name="analyze_review_authenticity",
            description="Analyze review for authenticity and legitimacy",
            inputSchema={
                "type": "object",
                "properties": {
                    "review_data": {
                        "type": "object",
                        "description": "Complete review data for authenticity analysis"
                    },
                    "user_history": {
                        "type": "object",
                        "description": "User's review history and account information"
                    }
                },
                "required": ["review_data"]
            }
        ),
        Tool(
            name="update_spam_patterns",
            description="Update spam detection patterns with new data",
            inputSchema={
                "type": "object",
                "properties": {
                    "new_patterns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "New spam patterns to add"
                    },
                    "pattern_type": {
                        "type": "string",
                        "enum": ["regex", "keyword", "phrase"],
                        "description": "Type of patterns being added"
                    }
                },
                "required": ["new_patterns", "pattern_type"]
            }
        )
    ]

@server.call_tool()
async def call_tool(request: CallToolRequest) -> CallToolResult:
    """Handle tool calls for review spam detection."""
    
    if request.name == "analyze_review_spam":
        review_data = request.arguments.get("review_data", {})
        context = request.arguments.get("context", {})
        
        # Create mock MCP packet for analysis
        mcp_packet = {
            "raw_data": review_data,
            "context": context,
            "timestamp": datetime.now().isoformat(),
            "sub_agent_results": {},
            "risk_scores": {}
        }
        
        result = spam_agent.detect_spam(mcp_packet)
        
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps({
                    "spam_score": result.spam_score,
                    "spam_signals": result.spam_signals,
                    "confidence": result.confidence,
                    "model_version": result.model_version,
                    "analysis_summary": {
                        "risk_level": "high" if result.spam_score > 0.7 else "medium" if result.spam_score > 0.4 else "low",
                        "primary_concerns": result.spam_signals[:3],
                        "recommendation": "block" if result.spam_score > 0.8 else "flag" if result.spam_score > 0.5 else "allow"
                    }
                }, indent=2)
            )]
        )
    
    elif request.name == "get_spam_patterns":
        pattern_category = request.arguments.get("pattern_category", "all")
        
        patterns = {
            "text": {
                "promotional_language": ["discount", "sale", "deal", "offer", "bargain", "cheap", "free"],
                "spam_phrases": spam_agent.known_spam_patterns,
                "suspicious_patterns": ["excessive_punctuation", "excessive_caps", "repeated_characters"]
            },
            "behavioral": {
                "review_patterns": ["duplicate_review", "short_positive_review", "review_bombing_pattern"],
                "user_patterns": ["suspicious_review_frequency", "new_account_high_activity"],
                "timing_patterns": ["unusual_posting_time", "coordinated_reviews"]
            },
            "temporal": {
                "time_based": ["off_hours_posting", "burst_activity", "coordinated_timing"],
                "frequency": ["high_review_rate", "consistent_intervals", "suspicious_gaps"]
            }
        }
        
        if pattern_category == "all":
            result_patterns = patterns
        else:
            result_patterns = {pattern_category: patterns.get(pattern_category, {})}
        
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps({
                    "pattern_category": pattern_category,
                    "patterns": result_patterns
                }, indent=2)
            )]
        )
    
    elif request.name == "analyze_review_authenticity":
        review_data = request.arguments.get("review_data", {})
        user_history = request.arguments.get("user_history", {})
        
        # Create mock MCP packet
        mcp_packet = {
            "raw_data": review_data,
            "context": {"customer_history": user_history},
            "timestamp": datetime.now().isoformat()
        }
        
        authenticity_signals = spam_agent.analyze_review_authenticity(mcp_packet)
        text_signals = spam_agent.analyze_text_patterns(review_data.get("review_text", ""))
        
        # Calculate authenticity score (inverse of spam likelihood)
        total_signals = len(authenticity_signals) + len(text_signals)
        authenticity_score = max(0.0, 1.0 - (total_signals * 0.15))
        
        return CallToolResult(
            content=[TextContent(
                type="text",
                text=json.dumps({
                    "authenticity_score": authenticity_score,
                    "authenticity_signals": authenticity_signals,
                    "text_analysis_signals": text_signals,
                    "overall_assessment": {
                        "likely_authentic": authenticity_score > 0.6,
                        "confidence_level": "high" if len(authenticity_signals) < 2 else "medium" if len(authenticity_signals) < 4 else "low",
                        "recommendation": "approve" if authenticity_score > 0.7 else "review" if authenticity_score > 0.4 else "reject"
                    }
                }, indent=2)
            )]
        )
    
    elif request.name == "update_spam_patterns":
        new_patterns = request.arguments.get("new_patterns", [])
        pattern_type = request.arguments.get("pattern_type", "regex")
        
        try:
            if pattern_type == "regex":
                # Validate regex patterns
                for pattern in new_patterns:
                    re.compile(pattern)
                spam_agent.known_spam_patterns.extend(new_patterns)
            else:
                # For keyword/phrase patterns, convert to regex
                regex_patterns = []
                for pattern in new_patterns:
                    if pattern_type == "keyword":
                        regex_patterns.append(rf"\b{re.escape(pattern)}\b")
                    else:  # phrase
                        regex_patterns.append(re.escape(pattern))
                spam_agent.known_spam_patterns.extend(regex_patterns)
            
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps({
                        "status": "success",
                        "message": f"Added {len(new_patterns)} new {pattern_type} patterns",
                        "total_patterns": len(spam_agent.known_spam_patterns),
                        "timestamp": datetime.now().isoformat()
                    }, indent=2)
                )]
            )
            
        except Exception as e:
            return CallToolResult(
                content=[TextContent(
                    type="text",
                    text=json.dumps({
                        "status": "error",
                        "message": f"Error updating patterns: {str(e)}"
                    }, indent=2)
                )]
            )
    
    else:
        raise ValueError(f"Unknown tool: {request.name}")

async def main():
    """Main entry point"""
    # Initialize spam agent
    await spam_agent.initialize()
    
    # Start Kafka consumer in background
    kafka_task = asyncio.create_task(spam_agent.run_kafka_consumer())
    
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