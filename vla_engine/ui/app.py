"""Streamlit interactive dashboard for the VLA Resource Discovery and Ranking Engine."""

import sys
from pathlib import Path

# Ensure package root is in sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import pandas as pd
import streamlit as st

from vla_engine.discovery.aggregator import DiscoveryAggregator
from vla_engine.evaluation.ranker import ResourceRanker
from vla_engine.models.query import ScoringWeights
from vla_engine.models.resource import Category
from vla_engine.parser.query_parser import QueryParser
from vla_engine.synthesis.blueprint_generator import BlueprintGenerator
from vla_engine.synthesis.formatter import ReportFormatter


# ----------------- PAGE CONFIG -----------------

st.set_page_config(
    page_title="VLA Resource Discovery & Ranking Engine",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------- CUSTOM CSS -----------------

st.markdown(
    """
    <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1E293B;
            margin-bottom: 0.2rem;
        }

        .sub-header {
            font-size: 1.05rem;
            color: #64748B;
            margin-bottom: 1.5rem;
        }

        .metric-card {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 12px;
        }

        .pillar-badge {
            background-color: #EFF6FF;
            color: #1D4ED8;
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------- SIDEBAR -----------------

with st.sidebar:
    st.header("⚙️ Evaluation Parameters")

    st.subheader("Scoring Weights Tuning")
    st.caption("Adjust importance for multi-dimensional ranking algorithm:")

    w_rel = st.slider(
        "Semantic & Task Relevance",
        0.0,
        1.0,
        0.35,
        0.05,
    )

    w_adopt = st.slider(
        "Community Adoption (Stars/DLs)",
        0.0,
        1.0,
        0.20,
        0.05,
    )

    w_repro = st.slider(
        "Reproducibility & Open License",
        0.0,
        1.0,
        0.20,
        0.05,
    )

    w_fresh = st.slider(
        "Freshness & Recency",
        0.0,
        1.0,
        0.15,
        0.05,
    )

    w_compat = st.slider(
        "Stack Ecosystem Compatibility",
        0.0,
        1.0,
        0.10,
        0.05,
    )

    weights = ScoringWeights(
        semantic_relevance=w_rel,
        community_adoption=w_adopt,
        reproducibility=w_repro,
        freshness=w_fresh,
        compatibility=w_compat,
    ).normalize()

    st.markdown("---")

    use_live_apis = st.checkbox(
        "Enable Live APIs (HF, GitHub, ArXiv)",
        value=True,
    )

    top_k = st.slider(
        "Top Candidates per Pillar",
        3,
        10,
        5,
    )


# ----------------- MAIN PANEL -----------------

st.markdown(
    '<div class="main-header">🤖 Intelligent VLA & Robotics Search Engine</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="sub-header">'
    "Autonomous discovery, multi-source ingestion, multi-factor scoring, "
    "and training blueprint synthesis for Vision-Language-Action models."
    "</div>",
    unsafe_allow_html=True,
)


# ----------------- SEARCH QUERY -----------------

query_input = st.text_area(
    "Search Query:",
    value="",
    height=90,
    placeholder=(
        "Enter your VLA or robotics resource requirement here..."
    ),
)


# ----------------- RUN BUTTON -----------------

col_btn, col_info = st.columns([1, 4])

with col_btn:
    run_btn = st.button(
        "🚀 Discover & Rank",
        type="primary",
        use_container_width=True,
    )


# ----------------- MAIN PIPELINE -----------------

if run_btn or "blueprint_cache" in st.session_state:

    if run_btn:

        if not query_input.strip():
            st.warning(
                "Please enter a VLA or robotics search query before running the engine."
            )
            st.stop()

        with st.spinner(
            "Analyzing intent, discovering multi-source resources, "
            "and computing composite rankings..."
        ):

            # 1. Parse and understand the query
            parser = QueryParser(default_weights=weights)

            parsed_query = parser.parse(
                query_input,
                custom_weights=weights,
            )

            # 2. Discover resources
            aggregator = DiscoveryAggregator(
                use_live_apis=use_live_apis
            )

            discovered_resources = aggregator.discover_all(
                parsed_query
            )

            # 3. Rank resources
            ranker = ResourceRanker()

            ranked_categories = ranker.rank_resources(
                discovered_resources,
                parsed_query,
                top_k_per_category=top_k,
            )

            # 4. Generate VLA blueprint
            generator = BlueprintGenerator()

            blueprint = generator.generate(
                ranked_categories,
                parsed_query,
            )

            # 5. Store results in session state
            st.session_state["blueprint_cache"] = blueprint
            st.session_state["parsed_query"] = parsed_query
            st.session_state["discovered_count"] = len(
                discovered_resources
            )

    else:
        blueprint = st.session_state["blueprint_cache"]
        parsed_query = st.session_state["parsed_query"]


    # ----------------- 1. QUERY INTENT -----------------

    st.markdown("### 1. Extracted Query Intent")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Target Tasks",
            ", ".join(blueprint.query.target_tasks)
            or "Object Manipulation",
        )

    with c2:
        st.metric(
            "Modalities",
            ", ".join(blueprint.query.modalities),
        )

    with c3:
        st.metric(
            "Embodiments",
            ", ".join(blueprint.query.target_embodiments)
            or "Multi-Embodiment Arms",
        )

    with c4:
        st.metric(
            "Compatibility",
            f"{blueprint.compatibility_score:.1f} / 100",
        )


    # ----------------- 2. SUGGESTED VLA STACK -----------------

    st.markdown("### 2. Suggested VLA Technical Stack")

    st.info(
        f"**Coherence Verdict:** "
        f"{blueprint.compatibility_verdict}"
    )

    if blueprint.top_stack:

        cols = st.columns(len(blueprint.top_stack))

        for idx, (cat, res) in enumerate(
            blueprint.top_stack.items()
        ):

            with cols[idx]:

                st.markdown(
                    '<div class="metric-card">',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f'<span class="pillar-badge">'
                    f'{cat.display_name}'
                    f'</span>',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"#### {res.name}"
                )

                st.markdown(
                    f"**Score:** `{res.final_score:.1f}/100` "
                    f"| `{res.license}`"
                )

                st.caption(
                    res.summary[:130] + "..."
                )

                links = []

                if res.github_url:
                    links.append(
                        f"[GitHub]({res.github_url})"
                    )

                if res.hf_url:
                    links.append(
                        f"[HuggingFace]({res.hf_url})"
                    )

                if res.paper_url:
                    links.append(
                        f"[Paper]({res.paper_url})"
                    )

                if links:
                    st.markdown(
                        " • ".join(links)
                    )

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True,
                )


    # ----------------- 3. INTERACTIVE DETAILS -----------------

    tab_leaderboard, tab_blueprint, tab_hardware, tab_charts = st.tabs(
        [
            "📊 Category Leaderboards",
            "🗺️ VLA Training Blueprint",
            "💻 Hardware Sizing",
            "📈 Comparative Charts",
        ]
    )


    # ----------------- LEADERBOARD -----------------

    with tab_leaderboard:

        st.markdown(
            "#### Ranked Resources by Category"
        )

        for cat, res_list in (
            blueprint.all_ranked_resources.items()
        ):

            st.markdown(
                f"##### {cat.display_name}"
            )

            data = []

            for rank, r in enumerate(
                res_list,
                1,
            ):

                sc = r.scores

                data.append(
                    {
                        "Rank": f"#{rank}",
                        "Name": r.name,
                        "Composite Score": (
                            sc.composite_score
                            if sc
                            else r.final_score
                        ),
                        "Relevance (35%)": (
                            sc.semantic_relevance
                            if sc
                            else 0
                        ),
                        "Adoption (20%)": (
                            sc.community_adoption
                            if sc
                            else 0
                        ),
                        "Reproducibility (20%)": (
                            sc.reproducibility
                            if sc
                            else 0
                        ),
                        "Freshness (15%)": (
                            sc.freshness
                            if sc
                            else 0
                        ),
                        "Compatibility (10%)": (
                            sc.compatibility
                            if sc
                            else 0
                        ),
                        "Stars": r.github_stars,
                        "HF Downloads": r.hf_downloads,
                        "License": r.license,
                        "URL": (
                            r.url
                            or r.github_url
                            or r.hf_url
                            or ""
                        ),
                    }
                )

            df = pd.DataFrame(data)

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
            )


    # ----------------- TRAINING BLUEPRINT -----------------

    with tab_blueprint:

        st.markdown(
            "#### End-to-End VLA Training & Deployment Roadmap"
        )

        for step in blueprint.pipeline_steps:

            with st.expander(
                f"Step {step.step_number}: {step.title}",
                expanded=True,
            ):

                st.write(step.description)

                col_s1, col_s2 = st.columns(2)

                with col_s1:

                    st.markdown(
                        f"**Recommended Tools:** "
                        f"{', '.join(step.recommended_tools)}"
                    )

                    st.markdown(
                        f"**Inputs:** {step.inputs}"
                    )

                with col_s2:

                    st.markdown(
                        f"**Outputs:** {step.outputs}"
                    )

                    st.markdown(
                        f"**Hardware Requirement:** "
                        f"{step.hardware_notes}"
                    )


    # ----------------- HARDWARE -----------------

    with tab_hardware:

        st.markdown(
            "#### Hardware Infrastructure Requirements"
        )

        hw_data = [
            {
                "Infrastructure Tier": k,
                "Specification": v,
            }
            for k, v in blueprint.hardware_summary.items()
        ]

        st.table(
            pd.DataFrame(hw_data)
        )


    # ----------------- COMPARATIVE CHARTS -----------------

    with tab_charts:

        st.markdown(
            "#### Comparative Score Distribution Across Pillars"
        )

        all_chart_data = []

        for cat, res_list in (
            blueprint.all_ranked_resources.items()
        ):

            for r in res_list[:3]:

                if r.scores:

                    all_chart_data.append(
                        {
                            "Resource": r.name,
                            "Category": cat.display_name,
                            "Composite Score": (
                                r.scores.composite_score
                            ),
                            "Relevance": (
                                r.scores.semantic_relevance
                            ),
                            "Adoption": (
                                r.scores.community_adoption
                            ),
                            "Reproducibility": (
                                r.scores.reproducibility
                            ),
                        }
                    )

        if all_chart_data:

            chart_df = pd.DataFrame(
                all_chart_data
            )

            st.bar_chart(
                chart_df,
                x="Resource",
                y="Composite Score",
                color="Category",
            )


    # ----------------- EXPORT -----------------

    st.markdown("---")

    st.markdown(
        "### 📥 Export Evaluation Artifacts"
    )

    c_exp1, c_exp2 = st.columns(2)

    with c_exp1:

        md_content = ReportFormatter.to_markdown(
            blueprint
        )

        st.download_button(
            label="📄 Download Markdown Dossier (report.md)",
            data=md_content,
            file_name="vla_training_blueprint.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with c_exp2:

        json_content = ReportFormatter.to_json(
            blueprint
        )

        st.download_button(
            label="💾 Download JSON Evaluation Data (report.json)",
            data=json_content,
            file_name="vla_training_blueprint.json",
            mime="application/json",
            use_container_width=True,
        )
