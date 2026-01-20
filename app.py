import streamlit as st
import json
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

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
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .candidate-card {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .score-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1.2rem;
    }
    .score-high {
        background: #4caf50;
        color: white;
    }
    .score-medium {
        background: #ff9800;
        color: white;
    }
    .score-low {
        background: #f44336;
        color: white;
    }
    .skill-tag {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        margin: 0.2rem;
        border-radius: 15px;
        font-size: 0.85rem;
    }
    .skill-matched {
        background: #e8f5e9;
        color: #2e7d32;
        border: 1px solid #4caf50;
    }
    .skill-missing {
        background: #ffebee;
        color: #c62828;
        border: 1px solid #f44336;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_results():
    if Path("results_hybrid.json").exists():
        with open("results_hybrid.json", "r") as f:
            return json.load(f)
    with open("results_llm_only.json", "r") as f:
        return json.load(f)

@st.cache_data
def load_ground_truth():
    with open("data/labeled_dataset.json", "r") as f:
        return json.load(f)

@st.cache_data
def load_resume(resume_id):
    resume_path = f"data/resumes/{resume_id}.txt"
    if Path(resume_path).exists():
        with open(resume_path, "r") as f:
            return f.read()
    return "Resume not found"

# Helper functions
def get_score_class(score):
    if score >= 0.7:
        return "score-high"
    elif score >= 0.4:
        return "score-medium"
    else:
        return "score-low"

def get_tier_label(score):
    if score >= 0.7:
        return "🟢 Good Match"
    elif score >= 0.4:
        return "🟡 Partial Match"
    else:
        return "🔴 Poor Match"

# Main app
def main():
    # Header
    st.markdown('<div class="main-header">🎯 AI Resume Matcher</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ema AI Applications Engineer - Candidate Evaluation Dashboard</div>', unsafe_allow_html=True)
    
    # Load data
    results = load_results()
    ground_truth = load_ground_truth()
    
    # Sidebar
    st.sidebar.title("🚀 Deployment & Config")
    
    # API Key Management
    api_key_input = st.sidebar.text_input("Enter Groq API Key", type="password", help="Needed to run new evaluations. If left blank, we will try to use the environment variable.")
    
    if api_key_input:
        import os
        os.environ["GROQ_API_KEY"] = api_key_input
    
    if not os.getenv("GROQ_API_KEY"):
        st.sidebar.warning("⚠️ No Groq API Key found. You can view existing results, but cannot run new evaluations.")

    st.sidebar.markdown("---")
    st.sidebar.title("📊 Dashboard Controls")
    
    view_mode = st.sidebar.radio(
        "Select View",
        ["Overview", "Detailed Rankings", "Individual Candidate", "Score Analysis"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Quick Stats")
    st.sidebar.metric("Total Candidates", len(results))
    
    # Calculate scores safely
    scores = [r.get('final_score', r.get('score', 0)) for r in results]
    avg_score = sum(scores) / len(results) if results else 0
    st.sidebar.metric("Avg Score", f"{avg_score:.2f}")
    
    top_score = max(scores) if scores else 0
    st.sidebar.metric("Top Score", f"{top_score:.2f}")
    
    # Main content based on view mode
    if view_mode == "Overview":
        show_overview(results, ground_truth)
    elif view_mode == "Detailed Rankings":
        show_detailed_rankings(results)
    elif view_mode == "Individual Candidate":
        show_individual_candidate(results)
    else:
        show_score_analysis(results, ground_truth)

def show_overview(results, ground_truth):
    st.header("📊 Overview")
    
    # Top 3 candidates
    col1, col2, col3 = st.columns(3)
    
    for i, (col, candidate) in enumerate(zip([col1, col2, col3], results[:3])):
        with col:
            rank_emoji = ["🥇", "🥈", "🥉"][i]
            st.markdown(f"### {rank_emoji} Rank {i+1}")
            st.markdown(f"**{candidate['id'].replace('_', ' ').title()}**")
            score = candidate.get('final_score', candidate.get('score', 0))
            score_class = get_score_class(score)
            st.markdown(f'<div class="score-badge {score_class}">{score:.2f}</div>', 
                       unsafe_allow_html=True)
            st.markdown(f"_{get_tier_label(score)}_")
    
    st.markdown("---")
    
    # Score distribution chart
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Score Distribution")
        fig = go.Figure()
        
        colors = [get_score_class(r['score']) for r in results]
        color_map = {
            'score-high': '#4caf50',
            'score-medium': '#ff9800',
            'score-low': '#f44336'
        }
        
        fig.add_trace(go.Bar(
            x=[r['id'].replace('res_', '').replace('_', ' ').title() for r in results],
            y=[r.get('final_score', r.get('score', 0)) for r in results],
            marker_color=[color_map[c] for c in colors],
            text=[f"{r.get('final_score', r.get('score', 0)):.2f}" for r in results],
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Candidate Scores",
            xaxis_title="Candidate",
            yaxis_title="Score",
            yaxis_range=[0, 1],
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Tier Distribution")
        
        tier_counts = {
            "Good (≥0.7)": sum(1 for r in results if r.get('final_score', r.get('score', 0)) >= 0.7),
            "Partial (0.4-0.7)": sum(1 for r in results if 0.4 <= r.get('final_score', r.get('score', 0)) < 0.7),
            "Poor (<0.4)": sum(1 for r in results if r.get('final_score', r.get('score', 0)) < 0.4)
        }
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=list(tier_counts.keys()),
            values=list(tier_counts.values()),
            marker_colors=['#4caf50', '#ff9800', '#f44336'],
            hole=0.4
        )])
        
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)

def show_detailed_rankings(results):
    st.header("🏆 Detailed Rankings")
    
    for i, candidate in enumerate(results, 1):
        with st.container():
            st.markdown(f'<div class="candidate-card">', unsafe_allow_html=True)
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"### #{i} - {candidate['id'].replace('_', ' ').title()}")
                st.markdown(f"**{get_tier_label(candidate['score'])}**")
            
            with col2:
                score = candidate.get('final_score', candidate.get('score', 0))
                score_class = get_score_class(score)
                st.markdown(f'<div class="score-badge {score_class}">{score:.2f}</div>', 
                           unsafe_allow_html=True)
            
            # Hybrid Breakdown
            if 'components' in candidate:
                c = candidate['components']
                col_h1, col_h2 = st.columns(2)
                with col_h1:
                    st.markdown(f"**LLM Component ({int(c['llm']['weight']*100)}%)**")
                    st.progress(c['llm']['score'])
                    st.caption(f"Raw Score: {c['llm']['score']:.2f}")
                with col_h2:
                    st.markdown(f"**Deterministic ({int(c['deterministic']['weight']*100)}%)**")
                    st.progress(c['deterministic']['score'])
                    st.caption(f"Raw Score: {c['deterministic']['score']:.3f} | Exp: {c['deterministic']['years_experience']}y")

            # Reasoning
            st.markdown("#### 💭 AI Assessment")
            reasoning = candidate.get('components', {}).get('llm', {}).get('reasoning', candidate.get('reasoning', ''))
            st.info(reasoning)
            
            # Skills
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### ✅ Matched Skills")
                if candidate['matched_skills']:
                    skills_html = "".join([
                        f'<span class="skill-tag skill-matched">{skill}</span>'
                        for skill in candidate['matched_skills']
                    ])
                    st.markdown(skills_html, unsafe_allow_html=True)
                else:
                    st.markdown("_No matched skills identified_")
            
            with col2:
                st.markdown("#### ❌ Missing Skills")
                if candidate['missing_skills']:
                    skills_html = "".join([
                        f'<span class="skill-tag skill-missing">{skill}</span>'
                        for skill in candidate['missing_skills']
                    ])
                    st.markdown(skills_html, unsafe_allow_html=True)
                else:
                    st.markdown("_No critical gaps identified_")
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("")

def show_individual_candidate(results):
    st.header("👤 Individual Candidate Analysis")
    
    candidate_names = [r['id'].replace('_', ' ').title() for r in results]
    selected = st.selectbox("Select Candidate", candidate_names)
    
    # Find selected candidate
    selected_id = selected.lower().replace(' ', '_')
    candidate = next((r for r in results if r['id'] == selected_id), None)
    
    if candidate:
        # Header with score
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"## {selected}")
            score = candidate.get('final_score', candidate.get('score', 0))
            st.markdown(f"**{get_tier_label(score)}**")
        with col2:
            score_class = get_score_class(score)
            st.markdown(f'<div class="score-badge {score_class}" style="font-size: 2rem; padding: 1rem 2rem;">{score:.2f}</div>', 
                       unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["📝 Resume", "💭 Analysis", "📊 Skills Breakdown"])
        
        with tab1:
            resume_text = load_resume(candidate['id'])
            st.text_area("Resume Content", resume_text, height=400)
        
        with tab2:
            st.markdown("### Decision Reasoning")
            reasoning = candidate.get('components', {}).get('llm', {}).get('reasoning', candidate.get('reasoning', ''))
            st.info(reasoning)
            
            st.markdown("### Score Breakdown")
            score = candidate.get('final_score', candidate.get('score', 0))
            st.progress(score)
            st.caption(f"Final Score: {score:.2f} / 1.00")
            
            if 'components' in candidate:
                cols = st.columns(2)
                with cols[0]:
                    st.write(f"LLM Contribution: {candidate['components']['llm']['weighted_contribution']}")
                with cols[1]:
                    st.write(f"Determ. Contribution: {candidate['components']['deterministic']['weighted_contribution']}")
                
                # Technical Breakdown
                if 'technical_breakdown' in candidate['components']['llm']:
                    st.markdown("#### 🛠️ Technical Scoring Breakdown")
                    tb = candidate['components']['llm']['technical_breakdown']
                    
                    col_b1, col_b2, col_b3 = st.columns(3)
                    with col_b1:
                        st.metric("Skill Alignment", f"{tb.get('skill_alignment', 0):.2f}")
                    with col_b2:
                        st.metric("Experience Depth", f"{tb.get('experience_depth', 0):.2f}")
                    with col_b3:
                        st.metric("Domain Fit", f"{tb.get('domain_fit', 0):.2f}")
                    
                    st.caption("These scores represent the LLM's granular assessment before weighting.")
        
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### ✅ Matched Skills")
                if candidate['matched_skills']:
                    for skill in candidate['matched_skills']:
                        st.markdown(f"- {skill}")
                else:
                    st.markdown("_No matched skills_")
            
            with col2:
                st.markdown("### ❌ Missing Skills")
                if candidate['missing_skills']:
                    for skill in candidate['missing_skills']:
                        st.markdown(f"- {skill}")
                else:
                    st.markdown("_No missing skills_")

def show_score_analysis(results, ground_truth):
    st.header("📈 Score Analysis")
    
    # Create comparison dataframe
    gt_map = {r['id']: r['label'] for r in ground_truth['resumes']}
    
    comparison_data = []
    for r in results:
        current_score = r.get('final_score', r.get('score', 0))
        comparison_data.append({
            'Candidate': r['id'].replace('res_', '').replace('_', ' ').title(),
            'LLM Score': current_score,
            'Ground Truth': gt_map.get(r['id'], 0.0)
        })
    
    # Score comparison chart
    st.subheader("Engine Score vs Ground Truth")
    
    fig = go.Figure()
    
    candidates = [d['Candidate'] for d in comparison_data]
    
    fig.add_trace(go.Bar(
        name='Engine Score (Hybrid)',
        x=candidates,
        y=[d['LLM Score'] for d in comparison_data],
        marker_color='#1f77b4'
    ))
    
    fig.add_trace(go.Bar(
        name='Ground Truth',
        x=candidates,
        y=[d['Ground Truth'] for d in comparison_data],
        marker_color='#ff7f0e'
    ))
    
    fig.update_layout(
        barmode='group',
        yaxis_range=[0, 1],
        height=500,
        xaxis_title="Candidate",
        yaxis_title="Score"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis
    st.subheader("📊 Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Top-K accuracy
        top_k = 3
        top_llm = set([r['id'] for r in results[:top_k]])
        top_gt = set([r['id'] for r in sorted(ground_truth['resumes'], key=lambda x: x['label'], reverse=True)[:top_k]])
        accuracy = len(top_llm & top_gt) / top_k * 100
        st.metric(f"Top-{top_k} Accuracy", f"{accuracy:.0f}%")
    
    with col2:
        # Good candidates in top positions
        good_in_top = sum(1 for r in results[:3] if gt_map.get(r['id'], 0) >= 0.7)
        st.metric("Good Matches in Top 3", f"{good_in_top}/2")
    
    with col3:
        # Average score difference
        avg_diff = sum(abs(r.get('final_score', r.get('score', 0)) - gt_map.get(r['id'], 0)) for r in results) / len(results)
        st.metric("Avg Score Difference", f"{avg_diff:.2f}")

if __name__ == "__main__":
    main()
