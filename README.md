# Resume Matcher V3: Sequential Hybrid Approach

> **Version 3 of 3** — Combining deterministic grounding with LLM reasoning

## Overview

V3 addresses the reliability issues of V1 (LLM hallucination) and the rigidity of V2 (pattern-only matching) through **Sequential Enrichment**.

The LLM receives verified facts as context, constraining its reasoning to grounded information.

---

## Architecture: Sequential Enrichment

```
Resume.txt
    │
    ▼
┌──────────────────────┐
│  DETERMINISTIC STAGE │ ◀── Stage 1: Regex & Embeddings
│  (Fact Extraction)   │     Extract Years, Skills (Ground Truth)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  LLM REASONING STAGE │ ◀── Stage 2: Grounded LLM Call
│  (Nuance & Context)  │     "Here are the facts. Now evaluate fit."
└──────────┬───────────┘
           │
           ▼
   Final Hybrid Score
   (60% LLM + 40% Det)
```

---

## Performance Comparison

| Metric | V1 (Pure LLM) | V2 (Deterministic) | V3 (Hybrid) | Best |
|--------|---------------|---------------------|-------------|------|
| **nDCG@3** | 0.837 | 0.828 | **0.845** | V3 |
| **Precision@1** | 1.000 | 1.000 | 1.000 | Tie |
| **Cost / 1k resumes** | ~$20.00 | $0.00 | ~$10.00 | V2 |
| **Latency / resume** | ~3.5s | <0.1s | ~0.7s | V2 |
| **Hallucination risk** | High | None | Low (grounded) | V2 |

---

## Key Advantages

1. **Grounded reasoning:** Verified facts (e.g., "7 years experience") in the prompt prevent the LLM from fabricating metrics.
2. **Semantic bridging:** V3 correctly maps "Prometheus" to "logging tools" (missed by V2) because the LLM bridges specific tools to generic JD requirements.
3. **Reduced API calls:** Single LLM call (vs. V1's two stages), reducing cost and latency by 50%.

---

## Usage

```bash
git checkout v3-hybrid
python evaluate_v3.py
```

---

## Conclusion

The architecture evolved from an opaque LLM pipeline (V1) to a fast deterministic baseline (V2), culminating in a **grounded hybrid approach (V3)** that balances accuracy, explainability, and cost. This represents the production-ready architecture.
