"""
Streamlit dashboard for interactive resume evaluation.
Two-Stage LLM Resume Matcher with configurable weights.
"""

import streamlit as st
import json
import plotly.graph_objects as go
from pathlib import Path
import os
from dotenv import load_dotenv
from src.matching_engine import TwoStageMatchingEngine
from src.utils import (
    load_job_description, 
    load_resumes, 
    load_ground_truth,
    calculate_ndcg_at_k,
    calculate_precision_at_k,
    calculate_recall_at_k
)

# Load environment
load_dotenv()

# Page config
st.set_page_config(
    page_title="Resume Matcher | Two-Stage LLM",
    page_icon="📄",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1f77b4; margin-bottom: 0.5rem; }
    .score-high { background: #4caf50; color: white; padding: 0.3rem 0.8rem; border-radius: 15px; }
    .score-medium { background: #ff9800; color: white; padding: 0.3rem 0.8rem; border-radius: 15px; }
    .score-low { background: #f44336; color: white; padding: 0.3rem 0.8rem; border-radius: 15px; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'results' not in st.session_state:
    # Try loading cached results
    if Path("results.json").exists():
        with open("results.json", "r") as f:
            st.session_state.results = json.load(f)
    else:
        st.session_state.results = None

if 'metrics' not in st.session_state:
    st.session_state.metrics = None


def get_tier_label(score):
    """Get tier label for score."""
    if score >= 0.7: return "Strong Match"
    if score >= 0.4: return "Partial Match"
    return "Weak Match"


def get_tier_color(score):
    """Get color for score tier."""
    if score >= 0.7: return "#4caf50"  # Green
    if score >= 0.4: return "#ff9800"  # Orange
    return "#f44336"  # Red


def main():
    # Title
    st.markdown('<div class="main-header">🎯 Two-Stage LLM Resume Matcher</div>', unsafe_allow_html=True)
    st.markdown("**Intelligent resume evaluation with explainable AI**")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key
        api_key = st.text_input("Groq API Key", type="password", value=os.getenv("GROQ_API_KEY", ""))
        if api_key:
            os.environ["GROQ_API_KEY"] = api_key
        
        st.divider()
        
        # Weight sliders
        st.subheader("📊 Scoring Weights")
        skill_weight = st.slider("Skill Match", 0.0, 1.0, 0.50, 0.05)
        exp_weight = st.slider("Experience Depth", 0.0, 1.0, 0.30, 0.05)
        domain_weight = st.slider("Domain Fit", 0.0, 1.0, 0.20, 0.05)
        
        total_weight = skill_weight + exp_weight + domain_weight
        if abs(total_weight - 1.0) < 0.01:
            st.success(f"✅ Total: {total_weight:.2f}")
        else:
            st.error(f"❌ Total: {total_weight:.2f} (must = 1.0)")
        
        st.divider()
        
        # File upload
        st.subheader("📁 Add Resumes")
        uploaded_files = st.file_uploader(
            "Upload .txt resumes", 
            type=["txt"], 
            accept_multiple_files=True
        )
        
        st.divider()
        
        # Evaluation button
        if st.button("🚀 Run Evaluation", type="primary", use_container_width=True):
            if not os.getenv("GROQ_API_KEY"):
                st.error("Please enter a Groq API Key!")
            elif abs(total_weight - 1.0) > 0.01:
                st.error("Weights must sum to 1.0!")
            else:
                with st.spinner("Evaluating resumes..."):
                    # Initialize engine
                    engine = TwoStageMatchingEngine()
                    engine.update_weights(skill_weight, exp_weight, domain_weight)
                    
                    # Load data
                    jd = load_job_description()
                    resumes = load_resumes()
                    
                    # Add uploaded resumes
                    for uploaded_file in uploaded_files:
                        text = uploaded_file.read().decode("utf-8")
                        name = uploaded_file.name.replace(".txt", "")
                        resumes.append({"id": name, "text": text})
                    
                    # Evaluate
                    results = engine.evaluate_batch(jd, resumes)
                    st.session_state.results = results
                    
                    # Calculate metrics
                    try:
                        ground_truth = load_ground_truth()
                        predicted = [r["final_score"] for r in results if r["id"] in ground_truth]
                        actual = [ground_truth[r["id"]] for r in results if r["id"] in ground_truth]
                        
                        if predicted and actual:
                            ndcg = calculate_ndcg_at_k(predicted, actual, k=3)
                            prec = calculate_precision_at_k(predicted, actual, k=1)
                            rec = calculate_recall_at_k(predicted, actual, k=3)
                            st.session_state.metrics = {
                                "nDCG@3": ndcg,
                                "Precision@1": prec,
                                "Recall@3": rec
                            }
                    except Exception as e:
                        st.warning(f"Could not calculate metrics: {e}")
                    
                    # Save results
                    with open("results.json", "w") as f:
                        json.dump(results, f, indent=2)
                    
                    st.success(f"✅ Evaluated {len(results)} resumes!")

    # Main content area
    if st.session_state.results is None:
        st.info("👈 Click **Run Evaluation** to start, or load cached results.")
        
        # Show JD preview
        with st.expander("📋 Job Description Preview"):
            try:
                jd = load_job_description()
                st.text(jd[:1000] + "..." if len(jd) > 1000 else jd)
            except:
                st.write("Job description not found.")
        return
    
    results = st.session_state.results
    
    # Metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("📊 Candidates", len(results))
    
    with col2:
        st.metric("🏆 Top Score", f"{results[0]['final_score']:.3f}")
    
    with col3:
        avg_score = sum(r['final_score'] for r in results) / len(results)
        st.metric("📈 Average", f"{avg_score:.3f}")
    
    with col4:
        if st.session_state.metrics:
            st.metric("🎯 nDCG@3", f"{st.session_state.metrics['nDCG@3']:.3f}")
        else:
            st.metric("🎯 nDCG@3", "N/A")
    
    with col5:
        avg_time = sum(r.get('processing_time_seconds', 0) for r in results) / len(results)
        st.metric("⏱️ Avg Time", f"{avg_time:.1f}s")
    
    st.divider()
    
    # Create tabs
    tab_overview, tab_rankings, tab_details, tab_jd = st.tabs([
        "📊 Overview", "🏆 Rankings", "🔍 Detailed Analysis", "📋 Job Description"
    ])
    
    with tab_overview:
        # Score distribution chart
        st.subheader("Score Distribution")
        
        fig = go.Figure()
        
        names = [r['candidate_name'] for r in results]
        scores = [r['final_score'] for r in results]
        colors = [get_tier_color(s) for s in scores]
        
        fig.add_trace(go.Bar(
            x=names,
            y=scores,
            marker_color=colors,
            text=[f"{s:.2f}" for s in scores],
            textposition='auto'
        ))
        
        fig.update_layout(
            xaxis_title="Candidate",
            yaxis_title="Final Score",
            yaxis_range=[0, 1.0],
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Dimension breakdown chart
        st.subheader("Dimension Scores (Top 5)")
        
        top_5 = results[:5]
        dims = ['skill_match', 'experience_depth', 'domain_fit']
        dim_labels = ['Skills', 'Experience', 'Domain']
        
        fig2 = go.Figure()
        
        for i, dim in enumerate(dims):
            dim_scores = []
            for r in top_5:
                ds = r.get('dimension_scores', {}).get(dim, {})
                dim_scores.append(ds.get('score', 0))
            
            fig2.add_trace(go.Bar(
                name=dim_labels[i],
                x=[r['candidate_name'] for r in top_5],
                y=dim_scores
            ))
        
        fig2.update_layout(
            barmode='group',
            yaxis_range=[0, 1.0],
            height=350,
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab_rankings:
        st.subheader("🏆 Candidate Rankings")
        
        for i, result in enumerate(results, 1):
            score = result['final_score']
            tier = get_tier_label(score)
            color = get_tier_color(score)
            
            with st.expander(f"#{i} | {result['candidate_name']} — {score:.3f} ({tier})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Dimension Scores:**")
                    dims = result.get('dimension_scores', {})
                    
                    for dim_name, dim_data in dims.items():
                        if isinstance(dim_data, dict) and 'score' in dim_data:
                            pretty_name = dim_name.replace('_', ' ').title()
                            st.metric(pretty_name, f"{dim_data['score']:.2f}")
                
                with col2:
                    st.markdown("**Parsed Data:**")
                    parsed = result.get('parsed_data', {})
                    st.metric("Total Experience", f"{parsed.get('total_years_experience', 0):.1f} years")
                    st.metric("Skills Extracted", len(parsed.get('skills', [])))
                    st.metric("Positions", len(parsed.get('experience', [])))
                
                st.markdown("---")
                
                # Skills
                matched = result.get('matched_skills', [])
                missing = result.get('missing_skills', [])
                
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    st.markdown(f"**✅ Matched Skills ({len(matched)}):**")
                    st.write(", ".join(matched) if matched else "None")
                with col_s2:
                    st.markdown(f"**❌ Missing Skills ({len(missing)}):**")
                    st.write(", ".join(missing) if missing else "None")
                
                # Overall assessment
                assessment = result.get('dimension_scores', {}).get('overall_assessment', '')
                if assessment:
                    st.info(f"**Assessment:** {assessment}")
    
    with tab_details:
        st.subheader("🔍 Individual Analysis")
        
        selected_name = st.selectbox(
            "Select Candidate", 
            [r['candidate_name'] for r in results]
        )
        
        result = next(r for r in results if r['candidate_name'] == selected_name)
        
        st.markdown(f"### {selected_name}")
        st.markdown(f"**Final Score: {result['final_score']:.3f}** ({get_tier_label(result['final_score'])})")
        
        # Sub-tabs
        detail_tabs = st.tabs(["📝 Breakdown", "🧠 Reasoning", "📄 Raw Data"])
        
        with detail_tabs[0]:
            st.markdown(result.get('score_breakdown', 'No breakdown available'))
        
        with detail_tabs[1]:
            dims = result.get('dimension_scores', {})
            for dim_name, dim_data in dims.items():
                if isinstance(dim_data, dict) and 'reasoning' in dim_data:
                    pretty_name = dim_name.replace('_', ' ').title()
                    st.markdown(f"**{pretty_name}:** {dim_data['reasoning']}")
        
        with detail_tabs[2]:
            st.json(result)
    
    with tab_jd:
        st.subheader("📋 Job Description")
        try:
            jd = load_job_description()
            st.text_area("Full Job Description", jd, height=500)
        except:
            st.error("Job description not found.")
    
    # Download button
    st.divider()
    st.download_button(
        label="⬇️ Download Results (JSON)",
        data=json.dumps(results, indent=2),
        file_name="evaluation_results.json",
        mime="application/json"
    )


if __name__ == "__main__":
    main()
