# 🧪 Reviewer's Guide: How to Test the Engine

If you are reviewing this project, here is how you can quickly test it, including adding your own resume to the evaluation.

## 1. Quick Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd ema-resume-ranker

# Setup virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements-minimal.txt
pip install streamlit plotly  # For the UI
```

## 2. Configuration
Create a `.env` file in the root directory:
```bash
GROQ_API_KEY=your_groq_api_key_here
LLM_PROVIDER=groq
LLM_MODEL=llama-3.3-70b-versatile
```

## 3. How to Test Your Own Resume
To see how the engine ranks *you* against the Ema AI Applications Engineer role:

1.  **Add your resume**: Save your resume as a `.txt` file in `data/resumes/res_yourname.txt`.
2.  **Run the Engine**:
    ```bash
    python demo_hybrid.py
    ```
    This will process all resumes in the folder, including yours, and output a new `results_hybrid.json`.
3.  **Launch the Dashboard**:
    ```bash
    streamlit run app.py
    ```
    - Navigate to **"Detailed Rankings"** to see your score and the AI's reasoning.
    - Navigate to **"Individual Candidate"** and select your name to see a full breakdown of matched/missing skills.

## 4. Key Files to Inspect
- `demo_hybrid.py`: The core logic (Hybrid Scorer + nDCG metric implementation).
- `README.md`: The technical justification and architectural overview.
- `EVALUATION_RESULTS.md`: The summary of how the engine performed against the human-labeled ground truth.
