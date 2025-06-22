"""
Event Router for TSCC Super Agent
Handles routing logic and event type definitions
"""

from enum import Enum
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

class EventType(Enum):
    LISTING = "listing"
    REVIEW = "review"
    RETURN = "return"
    PURCHASE = "purchase"
    USER_REGISTRATION = "user_registration"
    SELLER_ONBOARDING = "seller_onboarding"
    PAYMENT = "payment"
    REFUND = "refund"

class MarketplaceEvent(BaseModel):
    """Marketplace event model"""
    event_type: str
    product_id: Optional[str] = None
    seller_id: Optional[str] = None
    customer_id: Optional[str] = None
    user_id: Optional[str] = None
    transaction_id: Optional[str] = None
    content: Dict[str, Any]
    metadata: Dict[str, Any] = {}
    
    class Config:
        schema_extra = {
            "example": {
                "event_type": "listing",
                "product_id": "prod_123",
                "seller_id": "seller_456",
                "content": {
                    "title": "iPhone 13 Pro",
                    "description": "Brand new iPhone",
                    "price": 999.99,
                    "category": "Electronics"
                },
                "metadata": {
                    "ip_address": "192.168.1.1",
                    "user_agent": "Mozilla/5.0...",
                    "timestamp": "2025-01-01T12:00:00Z"
                }
            }
        }

class EventRouter:
    """Handles routing logic for different event types"""
    
    def __init__(self):
        # Define routing rules - which agents should process which event types
        self.routing_rules = {
            EventType.LISTING: ["fraud_detection", "counterfeit_detection"],
            EventType.REVIEW: ["review_spam", "fraud_detection"],
            EventType.RETURN: ["return_anomaly", "fraud_detection"],
            EventType.PURCHASE: ["fraud_detection"],
            EventType.USER_REGISTRATION: ["fraud_detection"],
            EventType.SELLER_ONBOARDING: ["fraud_detection", "counterfeit_detection"],
            EventType.PAYMENT: ["fraud_detection"],
            EventType.REFUND: ["fraud_detection", "return_anomaly"]
        }
        
        # Define Kafka topics for each sub-agent
        self.sub_agent_topics = {
            "fraud_detection": "fraud-detection-topic",
            "counterfeit_detection": "counterfeit-detection-topic",
            "review_spam": "review-spam-topic",
            "return_anomaly": "return-anomaly-topic"
        }
        
        # Define priority levels for different event types
        self.event_priorities = {
            EventType.LISTING: "medium",
            EventType.REVIEW: "low",
            EventType.RETURN: "medium",
            EventType.PURCHASE: "high",
            EventType.USER_REGISTRATION: "low",
            EventType.SELLER_ONBOARDING: "high",
            EventType.PAYMENT: "high",
            EventType.REFUND: "medium"
        }
        
        # Define timeout values for different event types (in seconds)
        self.event_timeouts = {
            EventType.LISTING: 60,
            EventType.REVIEW: 30,
            EventType.RETURN: 45,
            EventType.PURCHASE: 90,
            EventType.USER_REGISTRATION: 30,
            EventType.SELLER_ONBOARDING: 120,
            EventType.PAYMENT: 90,
            EventType.REFUND: 60
        }

    def get_routing_rules(self, event_type: EventType) -> List[str]:
        """Get the list of sub-agents that should process this event type"""
        agents = self.routing_rules.get(event_type, [])
        logger.debug(f"Routing {event_type.value} to agents: {agents}")
        return agents

    def get_agent_topic(self, agent_name: str) -> str:
        """Get the Kafka topic for a specific sub-agent"""
        topic = self.sub_agent_topics.get(agent_name)
        if not topic:
            raise ValueError(f"Unknown agent name: {agent_name}")
        return topic

    def get_event_priority(self, event_type: EventType) -> str:
        """Get the priority level for an event type"""
        return self.event_priorities.get(event_type, "medium")

    def get_event_timeout(self, event_type: EventType) -> int:
        """Get the timeout value for an event type"""
        return self.event_timeouts.get(event_type, 60)

    def should_route_to_agent(self, event_type: EventType, agent_name: str, context: Dict[str, Any] = None) -> bool:
        """
        Determine if an event should be routed to a specific agent
        Can include additional logic based on context
        """
        # Basic routing check
        if agent_name not in self.get_routing_rules(event_type):
            return False
        
        # Additional contextual routing logic
        if context:
            # Example: Skip counterfeit detection for trusted sellers
            if (agent_name == "counterfeit_detection" and 
                context.get("seller_history", {}).get("trust_score", 0) > 0.9):
                logger.info(f"Skipping counterfeit detection for trusted seller")
                return False
            
            # Example: Always route high-value transactions to fraud detection
            if (agent_name == "fraud_detection" and 
                event_type == EventType.PURCHASE and
                context.get("transaction_amount", 0) > 1000):
                logger.info(f"High-value transaction routed to fraud detection")
                return True
            
            # Example: Route returns with high return frequency to return anomaly detection
            if (agent_name == "return_anomaly" and 
                event_type == EventType.RETURN and
                context.get("customer_history", {}).get("return_frequency", 0) > 0.3):
                logger.info(f"High return frequency customer routed to return anomaly detection")
                return True
        
        return True

    def get_routing_config(self, event_type: EventType) -> Dict[str, Any]:
        """Get complete routing configuration for an event type"""
        return {
            "agents": self.get_routing_rules(event_type),
            "priority": self.get_event_priority(event_type),
            "timeout": self.get_event_timeout(event_type),
            "topics": [self.get_agent_topic(agent) for agent in self.get_routing_rules(event_type)]
        }

    def add_routing_rule(self, event_type: EventType, agents: List[str]):
        """Add or update routing rule for an event type"""
        self.routing_rules[event_type] = agents
        logger.info(f"Updated routing rule for {event_type.value}: {agents}")

    def remove_agent_from_routing(self, agent_name: str):
        """Remove an agent from all routing rules (e.g., when agent is down)"""
        for event_type, agents in self.routing_rules.items():
            if agent_name in agents:
                self.routing_rules[event_type] = [a for a in agents if a != agent_name]
                logger.warning(f"Removed {agent_name} from {event_type.value} routing")

    def get_agent_load_balancing_topic(self, agent_name: str, partition_key: str = None) -> str:
        """
        Get topic with load balancing considerations
        Could be extended to include partition keys for better distribution
        """
        base_topic = self.get_agent_topic(agent_name)
        
        # For now, return base topic
        # Future enhancement: implement partition-based load balancing
        if partition_key:
            # Could implement consistent hashing here
            pass
        
        return base_topic

    def validate_event(self, event: MarketplaceEvent) -> bool:
        """Validate that an event has required fields for routing"""
        try:
            # Check if event type is valid
            EventType(event.event_type)
            
            # Check required fields based on event type
            event_type = EventType(event.event_type)
            
            if event_type in [EventType.LISTING, EventType.PURCHASE]:
                if not event.product_id:
                    logger.error(f"Missing product_id for {event_type.value} event")
                    return False
            
            if event_type in [EventType.SELLER_ONBOARDING, EventType.LISTING]:
                if not event.seller_id:
                    logger.error(f"Missing seller_id for {event_type.value} event")
                    return False
            
            if event_type in [EventType.PURCHASE, EventType.RETURN, EventType.REVIEW]:
                if not event.customer_id:
                    logger.error(f"Missing customer_id for {event_type.value} event")
                    return False
            
            if event_type in [EventType.PAYMENT, EventType.REFUND]:
                if not event.transaction_id:
                    logger.error(f"Missing transaction_id for {event_type.value} event")
                    return False
            
            # Validate content is not empty
            if not event.content:
                logger.error("Event content cannot be empty")
                return False
            
            return True
            
        except ValueError as e:
            logger.error(f"Invalid event type: {event.event_type}")
            return False
        except Exception as e:
            logger.error(f"Event validation error: {e}")
            return False

    def get_event_routing_metrics(self) -> Dict[str, Any]:
        """Get metrics about event routing configuration"""
        agent_usage = {}
        for agents in self.routing_rules.values():
            for agent in agents:
                agent_usage[agent] = agent_usage.get(agent, 0) + 1
        
        return {
            "total_event_types": len(self.routing_rules),
            "total_agents": len(self.sub_agent_topics),
            "agent_usage_frequency": agent_usage,
            "routing_rules": {et.value: agents for et, agents in self.routing_rules.items()},
            "event_priorities": {et.value: priority for et, priority in self.event_priorities.items()},
            "event_timeouts": {et.value: timeout for et, timeout in self.event_timeouts.items()}
        }

    def get_recommended_agents(self, event_content: Dict[str, Any], event_type: EventType) -> List[str]:
        """
        Get recommended agents based on event content analysis
        This could be enhanced with ML-based recommendations
        """
        base_agents = self.get_routing_rules(event_type)
        recommended = base_agents.copy()
        
        # Content-based recommendations
        if event_type == EventType.LISTING:
            # Check for suspicious keywords that might indicate counterfeits
            suspicious_keywords = ["replica", "copy", "fake", "imitation", "knockoff"]
            title = event_content.get("title", "").lower()
            description = event_content.get("description", "").lower()
            
            if any(keyword in title or keyword in description for keyword in suspicious_keywords):
                if "counterfeit_detection" not in recommended:
                    recommended.append("counterfeit_detection")
                    logger.info("Added counterfeit detection due to suspicious keywords")
        
        elif event_type == EventType.REVIEW:
            # Check for review spam indicators
            review_text = event_content.get("review_text", "")
            if len(review_text) < 10 or review_text.count("!") > 5:
                if "review_spam" not in recommended:
                    recommended.append("review_spam")
                    logger.info("Added review spam detection due to suspicious review patterns")
        
        elif event_type == EventType.RETURN:
            # Check for high-value returns
            return_amount = event_content.get("return_amount", 0)
            if return_amount > 500:
                if "return_anomaly" not in recommended:
                    recommended.append("return_anomaly")
                    logger.info("Added return anomaly detection for high-value return")
        
        return recommended

# Utility functions for event processing
def create_routing_context(event: MarketplaceEvent, historical_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create routing context from event and historical data"""
    context = {
        "event_metadata": event.metadata,
        "content_summary": {
            "content_keys": list(event.content.keys()),
            "content_size": len(str(event.content))
        }
    }
    
    if historical_data:
        context.update(historical_data)
    
    # Add derived features
    if event.content.get("price"):
        context["price_category"] = categorize_price(event.content["price"])
    
    if event.content.get("description"):
        context["description_length"] = len(event.content["description"])
        context["description_word_count"] = len(event.content["description"].split())
    
    return context

def categorize_price(price: float) -> str:
    """Categorize price into ranges"""
    if price < 10:
        return "very_low"
    elif price < 50:
        return "low"
    elif price < 200:
        return "medium"
    elif price < 1000:
        return "high"
    else:
        return "very_high"

def extract_risk_indicators(event: MarketplaceEvent) -> List[str]:
    """Extract potential risk indicators from event content"""
    indicators = []
    
    # Check for common fraud indicators
    if event.event_type == "listing":
        title = event.content.get("title", "").lower()
        description = event.content.get("description", "").lower()
        
        # Price-related indicators
        price = event.content.get("price", 0)
        if price == 0:
            indicators.append("zero_price")
        elif price < 1:
            indicators.append("extremely_low_price")
        
        # Text-related indicators
        if "urgent" in title or "urgent" in description:
            indicators.append("urgency_language")
        
        if "limited time" in title or "limited time" in description:
            indicators.append("time_pressure")
        
        if len(description) < 20:
            indicators.append("short_description")
        
        # Suspicious patterns
        if title.isupper():
            indicators.append("all_caps_title")
    
    elif event.event_type == "review":
        review_text = event.content.get("review_text", "")
        rating = event.content.get("rating", 0)
        
        if rating == 5 and len(review_text) < 10:
            indicators.append("short_five_star_review")
        
        if review_text.count("!") > 3:
            indicators.append("excessive_exclamation")
    
    return indicators