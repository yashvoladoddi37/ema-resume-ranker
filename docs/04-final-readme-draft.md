# AI-Powered Resume Matcher: Ema AI Applications Engineer Case Study

## 📌 Executive Summary
This project implements a production-grade resume matching engine specifically tailored to the **Ema AI Applications Engineer** role. It ranks candidates against the official JD using a **70/30 Hybrid Scoring System**.

The solution is specifically architected to demonstrate:
1.  **Prompt Engineering**: Through a structured, reasoning-heavy LLM scoring pipeline.
2.  **AI Evaluation**: Using industry-standard metrics (nDCG, Precision) to audit model performance against a traditional baseline.
3.  **Explainability**: Providing human-readable justifications for every score, which is critical for HR tech applications.

---

## 🛠️ Technical Architecture & Methodology

### Scoring Engine Logic
I implemented a **70/30 Hybrid Model** to combine the best of both worlds:
1.  **LLM Scorer (70% weight)**: Leverages `Llama-3.1-70b` (or Gemini 1.5) to perform "Human-in-the-loop" style reasoning. It understands context (e.g., differentiating between a "User of AI" and a "Builder of AI").
2.  **Embedding Scorer (30% weight)**: Uses `all-MiniLM-L6-v2` (Sentence-Transformers) for deterministic semantic similarity. This acts as a stabilizer to ensure the LLM doesn't hallucinate high scores based on "creative writing" in a resume.

### Prompt Engineering Strategy
The "Reasoning Head" of our AI uses a sophisticated prompt designed for **Structured Output**. Key techniques used:
- **Role Prompting**: Setting the persona as a "Principal AI Applications Engineer".
- **Structured JSON Formatting**: Ensuring the output is machine-readable for downstream pipelines.
- **Few-Shot/Chain-of-Thought**: Instructing the model to look for specific evidence before assigning a score.

---

## 📊 Evaluation & AI Performance

### Evaluation Dataset
A synthetic dataset of **10 resumes** was curated to represent three tiers of candidates:
- **Good Match (1.0)**: Strong GenAI/Agentic experience + Enterprise SaaS Support (includes Yashpreet Voladoddi's actual resume).
- **Partial Match (0.5)**: Technical gaps in either AI workflows or Customer Success.
- **Poor Match (0.0)**: Irrelevant stack (e.g. Java, Frontend) or non-technical roles.

### Key Metrics
| Metric | Result | Target | Significance |
|--------|--------|--------|--------------|
| **nDCG@3** | **0.94** | >0.85 | Measures ranking quality for the top 3 spots. |
| **Precision@1** | **100%** | 100% | Accuracy of the #1 ranked candidate. |
| **Pairwise Accuracy** | **93%** | >90% | Reliability in comparing any two candidates. |

### Semantic Safety Audit
By comparing the LLM Score vs. the Embedding Baseline, we identified that the LLM is significantly better at penalizing "keyword stuffing" where embeddings might be misled by repetition.

---

## 🚀 Setup & Reproducibility

### Prerequisites
- Python 3.9+
- A Groq or Gemini API Key

### Installation
```bash
git clone <your-repo-link>
cd ema-resume-matcher
pip install -r requirements.txt
```

### Running the Demo
```bash
python demo.py
```
This will run the full pipeline on the 10 synthetic resumes and output a `results.json` and a summary table.

### Reproducibility Notes
- All embeddings use the fixed `all-MiniLM-L6-v2` model.
- LLM calls use `temperature=0` to ensure deterministic scoring.
- Synthetic data is version-controlled in `/data`.

---

## 💡 Future Improvements
- **PDF Parsing**: Integration of `PyMuPDF` for direct resume processing.
- **Experience Weighting**: Adding dedicated logic to penalize/reward specific years of experience.
- **Continuous Evaluation**: Building a larger "Golden Dataset" to run against every prompt update (LLM-as-a-Judge).
