# 🚀 Resume Matcher V3: Sequential Hybrid Approach

> **Version 3 of 3** — The "Best of Both Worlds" Architecture

## The Breakthrough

V3 solves the "Unreliable LLM" problem of V1 and the "Rigid Regex" problem of V2 by using **Sequential Enrichment**.

Instead of letting the LLM wander through raw text, we **anchor** it with verified facts.

---

## 🏗️ Architecture: Sequential Enrichment

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

## 📊 Final Performance Comparison

| Metric | V1 (Pure LLM) | V2 (Det) | V3 (Hybrid) | Winner |
|--------|---------------|----------|-------------|--------|
| **nDCG@3** | 0.837 | 0.828 | **0.845** | **V3** 🏆 |
| **Precision@1** | 1.000 | 1.000 | 1.000 | **Tie** |
| **Cost / 1k** | ~$20.00 | **$0.00** | ~$10.00 | **V2** |
| **Latency / Res**| ~3.5s | **<0.1s** | ~0.7s | **V2** |
| **Hallucination**| High | **Zero** | Low (Grounded) | **V2** |

---

## 🔎 Why V3 Wins

1. **Grounded Reasoning:** By providing "Verified Facts" (e.g., "7 years experience") in the LLM prompt, we prevent the LLM from hallucinating its own counts.
2. **Contextual Bridging:** V3 correctly matches "Prometheus" to "logging tools" (which V2 missed) because the LLM uses its semantic knowledge to bridge the gap between verified specific tools and generic JD requirements.
3. **Efficiency:** V3 uses only **one** LLM call (compared to V1's two), cutting costs and latency by 50%.

---

## 🚀 How to Run

```bash
git checkout v3-hybrid
python evaluate_v3.py
```

---

## 🏆 Conclusion

We have successfully evolved the pipeline from a fragile "Black Box" (V1) to a "Dumb but Fast" baseline (V2), and finally to a **"Smart & Auditable" Hybrid (V3)**. This is the production-ready architecture for Ema's recruitment engine.
