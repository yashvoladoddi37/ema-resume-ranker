# Streamlit Dashboard Guide

## 🚀 Quick Start

```bash
# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Launch the dashboard
streamlit run app.py
```

The dashboard will automatically open in your browser at `http://localhost:8501`

---

## 📊 Dashboard Features

### 1. Overview
- **Top 3 Candidates**: Medal podium with scores
- **Score Distribution Chart**: Bar chart showing all candidate scores
- **Tier Distribution**: Pie chart showing Good/Partial/Poor breakdown
- **Quick Stats**: Total candidates, average score, top score

### 2. Detailed Rankings
- **All 10 Candidates**: Ranked list with full details
- **LLM Reasoning**: Complete technical justification for each score
- **Matched Skills**: Green tags showing aligned competencies
- **Missing Skills**: Red tags showing critical gaps
- **Tier Labels**: Color-coded Good/Partial/Poor indicators

### 3. Individual Candidate
- **Resume Tab**: Full resume text
- **Analysis Tab**: LLM reasoning and score breakdown with progress bar
- **Skills Breakdown Tab**: Detailed matched and missing skills lists
- **Interactive Selection**: Dropdown to choose any candidate

### 4. Score Analysis
- **LLM vs Ground Truth**: Side-by-side bar chart comparison
- **Performance Metrics**:
  - Top-3 Accuracy
  - Good Matches in Top 3
  - Average Score Difference
- **Correlation Analysis**: Visual comparison of model performance

---

## 🎨 Visual Features

- **Color-Coded Scores**:
  - 🟢 Green (≥0.7): Good Match
  - 🟡 Orange (0.4-0.7): Partial Match
  - 🔴 Red (<0.4): Poor Match

- **Interactive Charts**: Hover for details, zoom, pan
- **Responsive Design**: Works on desktop and tablet
- **Clean UI**: Professional gradient cards and modern styling

---

## 💡 Tips

1. **Navigate Views**: Use the sidebar radio buttons
2. **Explore Candidates**: Click through individual profiles
3. **Compare Scores**: Check the Score Analysis view
4. **Export Data**: Results are in `results_llm_only.json`

---

## 🔧 Troubleshooting

**Port already in use?**
```bash
streamlit run app.py --server.port 8502
```

**Dependencies missing?**
```bash
pip install -r requirements-ui.txt
```

**Data not loading?**
- Ensure `results_llm_only.json` exists
- Run `python demo_minimal.py` first if needed
