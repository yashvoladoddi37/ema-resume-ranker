import streamlit as st
import json
import plotly.graph_objects as go
from pathlib import Path
import os
import time
from demo_hybrid import HybridEngine, run_formal_evaluation

# Page config
st.set_page_config(
    page_title="AI Resume Matcher - Ema",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1f77b4; margin-bottom: 0.5rem; }
    .candidate-card { border: 1px solid #e0e0e0; border_radius: 10px; padding: 1.5rem; margin: 1rem 0; background: #fdfdfd; }
    .score-badge { display: inline-block; padding: 0.5rem 1rem; border-radius: 20px; font-weight: 600; font-size: 1.2rem; }
    .score-high { background: #4caf50; color: white; }
    .score-medium { background: #ff9800; color: white; }
    .score-low { background: #f44336; color: white; }
    .skill-tag { display: inline-block; padding: 0.2rem 0.6rem; margin: 0.2rem; border-radius: 10px; font-size: 0.8rem; }
    .skill-matched { background: #e8f5e9; color: #2e7d32; border: 1px solid #4caf50; }
    .skill-missing { background: #ffebee; color: #c62828; border: 1px solid #f44336; }
</style>
""", unsafe_allow_html=True)

# State Management
if 'results' not in st.session_state:
    if Path("results_hybrid.json").exists():
        with open("results_hybrid.json", "r") as f:
            st.session_state.results = json.load(f)
    else:
        st.session_state.results = []

if 'metrics' not in st.session_state:
    st.session_state.metrics = {}

# Helper functions
def get_score_class(score):
    if score >= 0.7: return "score-high"
    if score >= 0.4: return "score-medium"
    return "score-low"

def get_tier_label(score):
    if score >= 0.7: return "🟢 Good Match"
    if score >= 0.4: return "🟡 Partial Match"
    return "🔴 Poor Match"

def load_jd():
    jd_path = "data/job_descriptions/ema_ai_apps_engineer.txt"
    if Path(jd_path).exists():
        return Path(jd_path).read_text()
    return "Job description not found."

def main():
    st.markdown('<div class="main-header">🎯 AI Resume Matcher</div>', unsafe_allow_html=True)
    st.markdown("_Sequential Hybrid Engine (60% LLM + 40% Deterministic)_")
    
    # Sidebar
    st.sidebar.title("💻 Engine Config")
    api_key = st.sidebar.text_input("Groq API Key", type="password")
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key
    
    st.sidebar.markdown("---")
    st.sidebar.title("📂 Upload Resumes")
    uploaded_files = st.sidebar.file_uploader("Add .txt resumes", type=["txt"], accept_multiple_files=True)
    
    if st.sidebar.button("🚀 Run Matching Engine", use_container_width=True):
        if not os.getenv("GROQ_API_KEY"):
            st.error("Please provide a Groq API Key in the sidebar.")
        else:
            with st.spinner("Analyzing resumes... (Including rate-limiting delays)"):
                engine = HybridEngine()
                jd_text = load_jd()
                
                # Load current resumes from data/resumes/
                all_resumes = []
                resume_dir = Path("data/resumes/")
                for f in resume_dir.glob("*.txt"):
                    all_resumes.append({"id": f.stem, "text": f.read_text()})
                
                # Add uploaded ones
                for uploaded_file in uploaded_files:
                    text = uploaded_file.read().decode("utf-8")
                    name = uploaded_file.name.replace(".txt", "").lower().replace(" ", "_")
                    all_resumes.append({"id": name, "text": text})
                
                # Run engine
                results = engine.rank_all(jd_text, all_resumes)
                st.session_state.results = results
                
                # Formal Evaluation if possible
                try:
                    with open("data/labeled_dataset.json", "r") as f:
                        gt_data = json.load(f)
                    metrics, _, _ = run_formal_evaluation(results, gt_data)
                    st.session_state.metrics = metrics
                except Exception as e:
                    st.warning(f"Could not calculate evaluation metrics: {e}")
                
                # Save results
                with open("results_hybrid.json", "w") as f:
                    json.dump(results, f, indent=2)
                st.success(f"Successfully evaluated {len(all_resumes)} resumes!")

    view_mode = st.sidebar.radio("Navigation", ["Overview", "Detailed Rankings", "Individual Analysis", "Job Description"])

    if view_mode == "Job Description":
        st.header("📄 Target Job Description")
        st.write(load_jd())

    elif view_mode == "Overview":
        if not st.session_state.results:
            st.info("No results yet. Run the engine from the sidebar.")
            return

        # Metrics Row
        if st.session_state.metrics:
            m = st.session_state.metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("nDCG@3", f"{m.get('ndcg3', 0):.3f}", help="Ranking quality vs ground truth")
            c2.metric("Precision@1", f"{m.get('precision1', 0)*100:.0f}%", help="Is top candidate a Good Match?")
            c3.metric("Recall@3", f"{m.get('recall3', 0)*100:.0f}%", help="Are all Good candidates in top 3?")
            c4.metric("Pairwise Accuracy", f"{m.get('pairwise_acc', 0)*100:.1f}%", help="How often is the better candidate ranked higher?")
        
        st.markdown("---")
        
        # Rankings Table
        st.subheader("🏆 Leaderboard")
        results = st.session_state.results
        
        # Plotly chart
        fig = go.Figure()
        names = [r['id'].replace('_', ' ').title() for r in results]
        scores = [r['final_score'] for r in results]
        colors = ['#4caf50' if s >= 0.7 else '#ff9800' if s >= 0.4 else '#f44336' for s in scores]
        
        fig.add_trace(go.Bar(x=names, y=scores, marker_color=colors, text=[f"{s:.2f}" for s in scores], textposition='auto'))
        fig.update_layout(title="Candidate Scores", yaxis_range=[0, 1], height=400)
        st.plotly_chart(fig, use_container_width=True)

    elif view_mode == "Detailed Rankings":
        for i, res in enumerate(st.session_state.results, 1):
            score = res['final_score']
            with st.expander(f"#{i} - {res['id'].replace('_', ' ').title()} ({score:.2f})"):
                st.markdown(f"### {get_tier_label(score)}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**LLM Assessment**")
                    st.info(res['components']['llm']['reasoning'])
                with col2:
                    st.markdown("**Deterministic Extraction**")
                    det = res['components']['deterministic']
                    st.write(f"Experience: {det.get('years_experience', 0)} years")
                    st.write(f"Skill Coverage: {det.get('required_coverage_pct', 0)}%")
                
                # Skills
                matched = res['components']['llm'].get('matched_skills', [])
                missing = res['components']['llm'].get('missing_skills', [])
                
                c_s1, c_s2 = st.columns(2)
                with c_s1:
                    st.markdown("**✅ Matched**")
                    st.write(", ".join(matched) if matched else "None identified")
                with c_s2:
                    st.markdown("**❌ Missing**")
                    st.write(", ".join(missing) if missing else "None identified")

    elif view_mode == "Individual Analysis":
        results = st.session_state.results
        if not results:
            st.info("Run engine first.")
            return
        
        selected_name = st.selectbox("Select Candidate", [r['id'].replace('_', ' ').title() for r in results])
        res = next(r for r in results if r['id'].replace('_', ' ').title() == selected_name)
        
        st.header(selected_name)
        s_c1, s_c2, s_c3 = st.columns(3)
        tb = res['components']['llm'].get('technical_breakdown', {})
        s_c1.metric("Skill Alignment", f"{tb.get('skill_alignment', 0):.2f}")
        s_c2.metric("Experience Depth", f"{tb.get('experience_depth', 0):.2f}")
        s_c3.metric("Domain Fit", f"{tb.get('domain_fit', 0):.2f}")
        
        tabs = st.tabs(["💡 AI Reasoning", "📊 Raw Components", "📝 Resume Text"])
        with tabs[0]:
            st.info(res['components']['llm']['reasoning'])
        with tabs[1]:
            st.json(res['components'])
        with tabs[2]:
            path = Path(f"data/resumes/{res['id']}.txt")
            if path.exists():
                st.text_area("Resume Content", path.read_text(), height=400)
            else:
                st.write("Original file not available in standard directory.")

if __name__ == "__main__":
    main()
