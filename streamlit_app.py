# streamlit_app.py
"""
Interactive Web Dashboard for Research Paper Recommendations
Run with: streamlit run streamlit_app.py
"""

import json
import pickle
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

sys.path.append(".")

from src.profile_builder import ResearchProfileBuilder
from src.recommender import ResearchRecommender

# Page config
st.set_page_config(
    page_title="Research Paper Recommender",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Force dark theme
st.markdown(
    """
<style>
    /* Dark mode styling */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }

    /* Fix text visibility */
    .stMarkdown, .stText, p, span, label {
        color: #ffffff !important;
    }

    /* Paper cards with dark theme */
    .paper-card {
        background-color: #1e2329;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        margin-bottom: 20px;
        border: 1px solid #2e3440;
    }

    .paper-card h3 {
        color: #ffffff !important;
    }

    .paper-card p {
        color: #e0e0e0 !important;
    }

    .relevance-score {
        font-size: 24px;
        font-weight: bold;
        color: #4CAF50;
    }

    .explanation-box {
        background-color: #2e3440;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        border: 1px solid #3e4450;
        color: #ffffff !important;
    }

    /* Sidebar styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background-color: #1e2329;
    }

    .css-1d391kg p, [data-testid="stSidebar"] p {
        color: #ffffff !important;
    }

    /* Metric styling */
    [data-testid="metric-container"] {
        background-color: #1e2329;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #2e3440;
    }

    [data-testid="metric-container"] label {
        color: #b0b0b0 !important;
    }

    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }

    /* Button styling */
    .stButton > button {
        background-color: #2e3440;
        color: #ffffff;
        border: 1px solid #3e4450;
    }

    .stButton > button:hover {
        background-color: #3e4450;
        border: 1px solid #4e5560;
    }

    /* Input fields */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > div {
        background-color: #2e3440;
        color: #ffffff;
        border: 1px solid #3e4450;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: #2e3440;
        color: #ffffff !important;
    }

    /* Links */
    a {
        color: #4CAF50 !important;
    }

    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }

    /* Radio buttons and checkboxes */
    .stRadio > label, .stCheckbox > label {
        color: #ffffff !important;
    }

    /* Success/Info/Warning messages */
    .stSuccess, .stInfo, .stWarning {
        background-color: #2e3440;
        color: #ffffff;
    }

    /* Plotly charts dark theme */
    .js-plotly-plot {
        background-color: #1e2329 !important;
    }
</style>
""",
    unsafe_allow_html=True,
)


# Load config
@st.cache_resource
def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


config = load_config()

# Sidebar
with st.sidebar:
    st.title("📚 Research Recommender")
    st.markdown(f"**Researcher**: {config['researcher']['name']}")

    st.markdown("---")

    # Navigation
    page = st.radio(
        "Navigate",
        [
            "📊 Dashboard",
            "🔍 Get Recommendations",
            "👤 Profile Analysis",
            "📈 Trends",
            "⚙️ Settings",
        ],
    )

    st.markdown("---")

    # Quick stats
    rec_files = list(Path("data/recommendations").glob("recommendations_*.json"))
    st.metric("Total Recommendations", len(rec_files))

    if rec_files:
        latest = max(rec_files, key=lambda x: x.stat().st_mtime)
        st.caption(f"Latest: {latest.stem.split('_')[1]}")

# Main content
if page == "📊 Dashboard":
    st.title("Research Paper Recommendations Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    # Load latest recommendations
    rec_files = list(Path("data/recommendations").glob("recommendations_*.json"))
    if rec_files:
        latest_file = max(rec_files, key=lambda x: x.stat().st_mtime)
        with open(latest_file, "r") as f:
            latest_recs = json.load(f)

        with col1:
            st.metric("Today's Papers", len(latest_recs))
        with col2:
            # Calculate average score properly
            scores = [
                r.get("relevance_score", r.get("similarity_score", 0.0))
                for r in latest_recs
            ]
            avg_score = sum(scores) / len(scores) if scores else 0.0
            st.metric("Avg Relevance", f"{avg_score:.2f}")
        with col3:
            categories = []
            for r in latest_recs:
                categories.extend(r.get("categories", []))
            st.metric("Categories", len(set(categories)))
        with col4:
            st.metric(
                "Sources", len(set(r.get("source", "unknown") for r in latest_recs))
            )

        st.markdown("---")

        # Today's recommendations
        st.subheader("📅 Today's Recommendations")

        for i, paper in enumerate(latest_recs[:10]):
            relevance_score = paper.get(
                "relevance_score", paper.get("similarity_score", 0.0)
            )

            with st.container():
                st.markdown(
                    f"""
                <div class="paper-card">
                    <h3 style="color: #ffffff;">{i+1}. {paper['title']}</h3>
                    <p style="color: #e0e0e0;"><strong>Authors:</strong> {', '.join(paper.get('authors', [])[:3])}{'...' if len(paper.get('authors', [])) > 3 else ''}</p>
                    <div class="relevance-score">Relevance: {relevance_score:.2f}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                if "explanation" in paper:
                    st.markdown(
                        f"""
                    <div class="explanation-box">
                        <strong style="color: #ffffff;">Why Relevant:</strong> <span style="color: #e0e0e0;">{paper['explanation']}</span>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                with st.expander("View Abstract"):
                    st.write(paper["abstract"])

                col1, col2 = st.columns([1, 5])
                with col1:
                    paper_url = paper.get("pdf_url", paper.get("url", "#"))
                    if paper_url and paper_url != "#":
                        st.markdown(f"[📄 Paper]({paper_url})")
                    else:
                        st.text("No link available")

                st.markdown("---")
    else:
        st.info("No recommendations yet. Click 'Get Recommendations' to generate.")

elif page == "🔍 Get Recommendations":
    st.title("Generate New Recommendations")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### Configuration")

        days_back = st.slider("Days to look back", 1, 7, 1)
        num_recommendations = st.slider("Number of recommendations", 5, 20, 10)
        use_rag = st.checkbox("Use RAG reranking", value=True)

        categories = st.multiselect(
            "arXiv Categories",
            ["cs.LG", "cs.AI", "cs.NE", "math.OC", "stat.ML", "math.NA"],
            default=config["sources"]["arxiv"]["categories"],
        )

    with col2:
        st.markdown("### Quick Presets")
        if st.button("🎯 Optimization Focus"):
            categories = ["math.OC", "cs.LG", "stat.ML"]
        if st.button("🧠 Neuro Focus"):
            categories = ["cs.NE", "cs.AI", "q-bio.NC"]
        if st.button("📐 Math Focus"):
            categories = ["math.NA", "math.OC", "math.CA"]

    if st.button("🚀 Generate Recommendations", type="primary"):
        with st.spinner("Fetching papers..."):
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Initialize recommender
            recommender = ResearchRecommender("config.yaml")

            status_text.text("Fetching papers from arXiv...")
            progress_bar.progress(20)

            # Get recommendations
            recommendations = recommender.get_daily_recommendations()

            status_text.text("Creating embeddings...")
            progress_bar.progress(50)

            status_text.text("Ranking with LLM...")
            progress_bar.progress(80)

            status_text.text("Generating report...")
            progress_bar.progress(100)

            st.success(f"✅ Generated {len(recommendations)} recommendations!")

            # Display results
            for i, paper in enumerate(recommendations):
                relevance_score = paper.get(
                    "relevance_score", paper.get("similarity_score", 0.0)
                )

                with st.container():
                    st.subheader(f"{i+1}. {paper['title']}")
                    st.write(f"**Relevance Score**: {relevance_score:.2f}")
                    if "explanation" in paper and paper["explanation"]:
                        st.write(
                            f"**Why Relevant**: {paper.get('explanation', 'Relevant to your research interests')}"
                        )

                    with st.expander("Details"):
                        st.write(
                            f"**Authors**: {', '.join(paper.get('authors', ['Unknown']))}"
                        )
                        st.write(
                            f"**Abstract**: {paper.get('abstract', 'No abstract available')}"
                        )
                        st.write(
                            f"**Categories**: {', '.join(paper.get('categories', []))}"
                        )
                        paper_url = paper.get("pdf_url", paper.get("url", ""))
                        if paper_url:
                            st.write(f"**Link**: {paper_url}")
                        else:
                            st.write("**Link**: Not available")

elif page == "👤 Profile Analysis":
    st.title("Research Profile Analysis")

    if st.button("🔄 Refresh Profile"):
        with st.spinner("Analyzing profile..."):
            builder = ResearchProfileBuilder()
            profile = builder.build_complete_profile()
            st.success("Profile updated!")

    # Load cached profile
    profile_path = Path("data/profile_cache/profile.pkl")
    if profile_path.exists():
        with open(profile_path, "rb") as f:
            profile = pickle.load(f)

        # Scholar metrics
        if profile.get("scholar_data"):
            st.subheader("📊 Google Scholar Metrics")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("h-index", profile["scholar_data"].get("h_index", 0))
            with col2:
                st.metric(
                    "Total Citations", profile["scholar_data"].get("total_citations", 0)
                )
            with col3:
                st.metric(
                    "Publications", len(profile["scholar_data"].get("papers", []))
                )

            # Publication timeline
            if profile["scholar_data"].get("papers"):
                papers_df = pd.DataFrame(profile["scholar_data"]["papers"])
                if "year" in papers_df.columns:
                    year_counts = papers_df["year"].value_counts().sort_index()

                    fig = px.bar(
                        x=year_counts.index,
                        y=year_counts.values,
                        labels={"x": "Year", "y": "Number of Papers"},
                        title="Publication Timeline",
                        template="plotly_dark",
                    )
                    fig.update_layout(
                        plot_bgcolor="#1e2329",
                        paper_bgcolor="#1e2329",
                        font_color="#ffffff",
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Top papers
            st.subheader("📝 Most Cited Papers")
            papers = profile["scholar_data"].get("papers", [])
            sorted_papers = sorted(
                papers, key=lambda x: x.get("citations", 0), reverse=True
            )[:5]

            for paper in sorted_papers:
                st.markdown(
                    f"- **{paper['title']}** ({paper.get('year', 'N/A')}) - {paper.get('citations', 0)} citations"
                )

        # Research interests word cloud
        st.subheader("🔍 Research Interests")
        interests = config["researcher"]["primary_interests"]
        st.write(", ".join(interests))

        # GitHub analysis
        if profile.get("github_interests"):
            st.subheader("⭐ GitHub Interests")
            topics = profile["github_interests"][:20]
            st.write(", ".join(topics))

elif page == "📈 Trends":
    st.title("Research Trends Analysis")

    # Load all recommendations
    rec_files = sorted(Path("data/recommendations").glob("recommendations_*.json"))

    if len(rec_files) > 1:
        # Trend analysis
        dates = []
        avg_scores = []
        paper_counts = []

        for file in rec_files[-30:]:  # Last 30 days
            with open(file, "r") as f:
                recs = json.load(f)

            date = datetime.strptime(file.stem.split("_")[1], "%Y%m%d")
            dates.append(date)

            if recs:
                avg_score = sum(r.get("relevance_score", 0) for r in recs) / len(recs)
                avg_scores.append(avg_score)
                paper_counts.append(len(recs))

        # Plot trends
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=avg_scores,
                mode="lines+markers",
                name="Avg Relevance Score",
                line=dict(color="blue", width=2),
            )
        )

        fig.update_layout(
            title="Average Relevance Score Trend",
            xaxis_title="Date",
            yaxis_title="Score",
            height=400,
            template="plotly_dark",
            plot_bgcolor="#1e2329",
            paper_bgcolor="#1e2329",
            font_color="#ffffff",
        )

        st.plotly_chart(fig, use_container_width=True)

        # Category distribution
        st.subheader("📊 Category Distribution")

        all_categories = []
        for file in rec_files[-7:]:  # Last week
            with open(file, "r") as f:
                recs = json.load(f)
            for rec in recs:
                all_categories.extend(rec.get("categories", []))

        if all_categories:
            cat_counts = pd.Series(all_categories).value_counts().head(10)

            fig = px.pie(
                values=cat_counts.values,
                names=cat_counts.index,
                title="Top Categories (Last 7 Days)",
                template="plotly_dark",
            )
            fig.update_layout(
                plot_bgcolor="#1e2329", paper_bgcolor="#1e2329", font_color="#ffffff"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(
            "Need more data for trend analysis. Generate recommendations for multiple days."
        )

elif page == "⚙️ Settings":
    st.title("Settings")

    # Load current config
    with open("config.yaml", "r") as f:
        current_config = yaml.safe_load(f)

    st.subheader("🔧 Configuration")

    # Edit interests
    st.markdown("### Research Interests")
    interests = st.text_area(
        "Primary Interests (one per line)",
        value="\n".join(current_config["researcher"]["primary_interests"]),
        height=150,
    )

    # LLM settings
    st.markdown("### LLM Settings")
    llm_provider = st.selectbox(
        "LLM Provider",
        ["ollama", "groq"],
        index=0 if current_config["llm"]["provider"] == "ollama" else 1,
    )

    if llm_provider == "groq":
        groq_key = st.text_input(
            "Groq API Key",
            value=current_config["llm"].get("groq_api_key", ""),
            type="password",
        )

    # Notification settings
    st.markdown("### Notifications")
    enable_email = st.checkbox(
        "Enable Email Notifications", value=current_config["notifications"]["enabled"]
    )

    if enable_email:
        email_sender = st.text_input(
            "Sender Email",
            value=current_config["notifications"]["email"].get("sender", ""),
        )
        email_password = st.text_input(
            "App Password",
            value=current_config["notifications"]["email"].get("password", ""),
            type="password",
        )
        email_recipient = st.text_input(
            "Recipient Email",
            value=current_config["notifications"]["email"].get("recipient", ""),
        )

    if st.button("💾 Save Settings"):
        # Update config
        current_config["researcher"]["primary_interests"] = interests.strip().split(
            "\n"
        )
        current_config["llm"]["provider"] = llm_provider

        if llm_provider == "groq":
            current_config["llm"]["groq_api_key"] = groq_key

        current_config["notifications"]["enabled"] = enable_email
        if enable_email:
            current_config["notifications"]["email"]["sender"] = email_sender
            current_config["notifications"]["email"]["password"] = email_password
            current_config["notifications"]["email"]["recipient"] = email_recipient

        # Save config
        with open("config.yaml", "w") as f:
            yaml.dump(current_config, f, default_flow_style=False)

        st.success("✅ Settings saved!")
        st.balloons()

# Footer
st.markdown("---")
st.markdown(
    "<center style='color: #ffffff;'>Built with ❤️ for Optimization & Fractional Calculus Research</center>",
    unsafe_allow_html=True,
)
