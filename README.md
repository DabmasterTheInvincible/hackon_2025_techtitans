# Trust & Safety Command Center (TSCC)
kjhghfcgcvbn
## Overview 

The Trust & Safety Command Center (TSCC) is an AI-powered platform designed to enhance marketplace trust by enabling real-time fraud detection, counterfeit screening, review authenticity verification, and return anomaly monitoring. Built for internal Trust & Safety analysts at Amazon, TSCC leverages large language models (LLMs), modular microservices, and streaming data pipelines to provide explainable, high-accuracy alerts with minimal noise.

# Team

## TechTitans

-Jenifer Shanmugasundaram

-Dileesha A

-Akshita Gupta

## Problem Statement

Trust & Safety teams face alert overload, inconsistent insights, and slow decision cycles when handling marketplace risks. TSCC addresses these challenges by providing an intelligent, real-time platform that aggregates agent outputs, applies dynamic rules, and generates LLM-powered investigation briefs to surface only high-priority cases.

## Features
Event Ingestion Pipeline: Real-time intake of marketplace events (e.g., listings, reviews, returns) using Kafka.

Super Agent Routing: Uses Model Context Protocol (MCP) to distribute event data to specialized sub-agents.

Sub-Agent Processing: Dedicated agents analyze fraud, counterfeit, review-spam, and return patterns.

LLM-Powered Briefs: Gemma-generated summaries for analysts with key risk metrics and context.

React Dashboard: UI to triage alerts, drill into MCP context, and take actions like "Confirm Fraud" or "Escalate".

Feedback Loop: Nightly retraining using analyst labels to improve model accuracy and reduce false positives.
