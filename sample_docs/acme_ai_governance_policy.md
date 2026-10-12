# Acme Corp AI Governance and Safety Policy

**Document ID:** POL-AI-2025-V2  
**Effective Date:** January 15, 2025  
**Version:** 2.4  
**Author:** AI Ethics and Governance Board  

---

## 1. Executive Summary & Purpose
This document establishes the mandatory operational safeguards, risk classification matrix, and compliance standards for all artificial intelligence and large language model (LLM) systems deployed within Acme Corp products and infrastructure.

## 2. Risk Classification Framework
Acme Corp categorizes all AI deployments into three distinct risk tiers:

### Tier 1: High Risk (Mission-Critical / High Impact)
- **Scope:** Customer-facing autonomous decision systems, healthcare triage assistants, automated financial underwriting, and employee performance evaluation models.
- **Mandatory Requirements:**
  - Mandatory Human-in-the-Loop (HITL) sign-off for any action affecting customer financial status or legal rights.
  - Quarterly algorithmic bias audits conducted by an independent certified third party.
  - Continuous output logging with a strict 365-day immutable audit log retention policy.
  - Automated toxicity and PII scanning on 100% of input prompts and model outputs.

### Tier 2: Medium Risk (Internal Productivity & Retrieval)
- **Scope:** Internal knowledge assistants (RAG systems), code generation copilots, document summarizers, and semantic search tools.
- **Mandatory Requirements:**
  - Automated retrieval grounding verification with confidence score thresholds of at least 0.35.
  - Mandatory inline source citation displaying originating document title, page number, and focal excerpt.
  - Bi-annual fairness review and hallucination rate benchmarking.
  - 90-day retention period for anonymized interaction telemetry logs.

### Tier 3: Low Risk (Creative & Non-Critical)
- **Scope:** Marketing copy brainstorming, generic template creation, and internal aesthetic design mockups.
- **Mandatory Requirements:**
  - Standard user notice that content was AI-assisted.
  - 30-day log retention.

## 3. Grounding, Citations, and Hallucination Avoidance
To prevent hallucinated facts and intellectual property risks:
1. **Evidence Rule:** Knowledge assistants must strictly decline to answer when relevant documents are absent or insufficient. The system must output a standard refusal message rather than conjecturing.
2. **Citation Requirement:** Every factual assertion must be attributed to a verified source chunk with cosine similarity >= 0.35.
3. **Data Freshness:** Vector indexes must be synchronized within 24 hours of any policy update.

## 4. Model Deprecation and Decommissioning
- Models deemed obsolete or superseded by newer foundational models must be phased out within 45 calendar days following the announcement.
- Legacy fine-tuned model checkpoints must be retained in encrypted cold storage for exactly 180 days prior to permanent cryptographic deletion.

## 5. Contact and Escalation
For urgent governance escalations or ethical concerns, contact the Ethics Committee at `ethics-board@acme-corp.internal` or incident channel `#incident-ai-safety`.
