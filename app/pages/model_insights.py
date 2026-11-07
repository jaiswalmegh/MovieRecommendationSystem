
from pathlib import Path
import html

import pandas as pd
import streamlit as st


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EVALUATION_PATH = PROJECT_ROOT / "artifacts" / "model_evaluation_results.csv"


# ============================================================
# HTML RENDERER
# ============================================================

def render_html(content):
    clean_html = "\n".join(
        line.lstrip()
        for line in content.strip().splitlines()
    )
    st.markdown(clean_html, unsafe_allow_html=True)


# ============================================================
# STYLES
# ============================================================

def inject_styles():
    st.markdown("""
    <style>
    .insight-hero {
        position:relative;
        overflow:hidden;
        padding:38px;
        border-radius:20px;
        border:1px solid rgba(201,244,90,.15);
        background:
            radial-gradient(circle at 90% 20%,rgba(80,231,210,.13),transparent 35%),
            linear-gradient(120deg,#172019,#0b100c);
        margin-bottom:28px;
    }

    .insight-eyebrow {
        color:#c9f45a;
        font-size:9px;
        font-weight:700;
        letter-spacing:3px;
        margin-bottom:13px;
    }

    .insight-heading {
        color:#f5f7ef;
        font:700 clamp(32px,4vw,48px)/1.1 'Space Grotesk',sans-serif;
        letter-spacing:-2px;
    }

    .insight-heading span { color:#c9f45a; }

    .insight-description {
        color:#a2ad9d;
        font-size:13px;
        line-height:1.8;
        max-width:650px;
        margin-top:12px;
    }

    .insight-section {
        color:#f4f5ef;
        font:700 22px 'Space Grotesk',sans-serif;
        margin:30px 0 5px;
    }

    .insight-caption {
        color:#879282;
        font-size:11px;
        margin-bottom:16px;
    }

    .insight-metric {
        background:linear-gradient(145deg,#171d16,#0d110e);
        border:1px solid rgba(255,255,255,.08);
        border-radius:15px;
        padding:19px;
        min-height:115px;
        transition:.25s ease;
    }

    .insight-metric:hover {
        transform:translateY(-4px);
        border-color:rgba(201,244,90,.3);
    }

    .insight-metric-label {
        color:#899285;
        font-size:9px;
        font-weight:700;
        letter-spacing:1.5px;
    }

    .insight-metric-value {
        color:#f5f7ef;
        font:700 27px 'Space Grotesk',sans-serif;
        margin-top:12px;
    }

    .insight-metric-note {
        color:#7c8876;
        font-size:9px;
        margin-top:5px;
    }

    .insight-panel {
        padding:20px;
        border-radius:15px;
        background:#111611;
        border:1px solid rgba(255,255,255,.07);
        margin-bottom:12px;
    }

    .insight-panel-title {
        color:#e9eee3;
        font:700 15px 'Space Grotesk',sans-serif;
        margin-bottom:12px;
    }

    .insight-label {
        color:#899486;
        font-size:10px;
        letter-spacing:1px;
    }

    .insight-value {
        color:#c9f45a;
        font:700 17px 'Space Grotesk',sans-serif;
    }

    .insight-status {
        display:inline-block;
        padding:6px 10px;
        border-radius:30px;
        border:1px solid rgba(201,244,90,.2);
        background:rgba(201,244,90,.07);
        color:#d9f889;
        font-size:9px;
        letter-spacing:1px;
    }

    .insight-flow {
        display:flex;
        flex-wrap:wrap;
        align-items:center;
        gap:8px;
        margin:18px 0;
    }

    .insight-flow-node {
        padding:12px 15px;
        background:#182018;
        border:1px solid rgba(201,244,90,.17);
        color:#eaf0e3;
        border-radius:9px;
        font-size:10px;
        font-weight:700;
        letter-spacing:.5px;
    }

    .insight-flow-arrow {
        color:#c9f45a;
        font-size:17px;
    }
    </style>
    """, unsafe_allow_html=True)


# ============================================================
# HELPERS
# ============================================================

def get_dataframe(obj, attribute):
    try:
        value = getattr(obj, attribute, None)

        if isinstance(value, pd.DataFrame):
            return value.copy()

    except Exception:
        pass

    return pd.DataFrame()


def get_attribute(obj, names, default=None):
    for name in names:
        try:
            value = getattr(obj, name, None)

            if value is not None:
                return value

        except Exception:
            continue

    return default


def get_shape(value):
    try:
        shape = value.shape
        return " × ".join(str(item) for item in shape)
    except Exception:
        return "Not available"


def format_number(value):
    try:
        return f"{int(value):,}"
    except (ValueError, TypeError):
        return "Not available"


def format_weight(value):
    try:
        return f"{float(value) * 100:.1f}%"
    except (ValueError, TypeError):
        return "Not available"


def find_column(df, candidates):
    lookup = {
        str(column).lower().strip(): column
        for column in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


# ============================================================
# MODEL COMPONENTS
# ============================================================

def get_components(model):
    content_model = get_attribute(
        model,
        ["content_model", "content_recommender"]
    )

    collaborative_model = get_attribute(
        model,
        ["collaborative_model", "collaborative_recommender"]
    )

    hybrid_model = get_attribute(
        model,
        ["hybrid_model", "hybrid_recommender"],
        model
    )

    return content_model, collaborative_model, hybrid_model


# ============================================================
# DATASET STATISTICS
# ============================================================

def get_dataset_stats(model):
    movies = get_dataframe(model, "movies")
    ratings = get_dataframe(model, "ratings")

    movie_id = find_column(
        movies,
        ["movie_id", "movieid", "id"]
    )

    rating_movie = find_column(
        ratings,
        ["movie_id", "movieid"]
    )

    user_id = find_column(
        ratings,
        ["user_id", "userid", "user"]
    )

    rating_col = find_column(
        ratings,
        ["rating", "score"]
    )

    users = (
        ratings[user_id].nunique()
        if user_id and not ratings.empty
        else 0
    )

    unique_rated_movies = (
        ratings[rating_movie].nunique()
        if rating_movie and not ratings.empty
        else 0
    )

    average_rating = (
        float(ratings[rating_col].mean())
        if rating_col and not ratings.empty
        else None
    )

    return {
        "movies": movies,
        "ratings": ratings,
        "movie_count": len(movies),
        "rating_count": len(ratings),
        "user_count": users,
        "unique_rated_movies": unique_rated_movies,
        "average_rating": average_rating,
        "movie_id": movie_id,
        "rating_movie": rating_movie,
        "user_id": user_id,
        "rating_col": rating_col
    }


# ============================================================
# METRIC CARD
# ============================================================

def render_metric(label, value, note):
    render_html(f"""
        <div class="insight-metric">
            <div class="insight-metric-label">
                {html.escape(str(label))}
            </div>

            <div class="insight-metric-value">
                {html.escape(str(value))}
            </div>

            <div class="insight-metric-note">
                {html.escape(str(note))}
            </div>
        </div>
    """)


# ============================================================
# OVERVIEW
# ============================================================

def render_overview(stats):
    render_html("""
        <div class="insight-section">
            The data behind the intelligence.
        </div>

        <div class="insight-caption">
            Dataset-level statistics used by the recommendation system.
        </div>
    """)

    values = [
        ("MOVIES", format_number(stats["movie_count"]), "MOVIE RECORDS"),
        ("RATINGS", format_number(stats["rating_count"]), "USER INTERACTIONS"),
        ("USERS", format_number(stats["user_count"]), "UNIQUE PROFILES"),
        (
            "AVERAGE RATING",
            f'{stats["average_rating"]:.2f}/5'
            if stats["average_rating"] is not None
            else "N/A",
            "DATASET AVERAGE"
        )
    ]

    columns = st.columns(4)

    for column, (label, value, note) in zip(columns, values):
        with column:
            render_metric(label, value, note)


# ============================================================
# HYBRID MODEL
# ============================================================

def render_hybrid_weights(model):
    content_weight = get_attribute(
        model,
        ["content_weight", "hybrid_content_weight"],
        None
    )

    collaborative_weight = get_attribute(
        model,
        ["collaborative_weight", "hybrid_collaborative_weight"],
        None
    )

    if content_weight is None or collaborative_weight is None:
        hybrid = get_attribute(
            model,
            ["hybrid_model", "hybrid_recommender"]
        )

        if hybrid is not None:
            content_weight = get_attribute(
                hybrid,
                ["content_weight", "hybrid_content_weight"],
                content_weight
            )

            collaborative_weight = get_attribute(
                hybrid,
                ["collaborative_weight", "hybrid_collaborative_weight"],
                collaborative_weight
            )

    render_html("""
        <div class="insight-section">
            Hybrid recommendation engine.
        </div>

        <div class="insight-caption">
            The system combines content similarity and collaborative signals.
        </div>
    """)

    col1, col2 = st.columns(2)

    with col1:
        render_html(f"""
            <div class="insight-panel">
                <div class="insight-label">CONTENT-BASED WEIGHT</div>
                <div class="insight-value">
                    {format_weight(content_weight)}
                </div>
            </div>
        """)

    with col2:
        render_html(f"""
            <div class="insight-panel">
                <div class="insight-label">COLLABORATIVE WEIGHT</div>
                <div class="insight-value">
                    {format_weight(collaborative_weight)}
                </div>
            </div>
        """)

    if content_weight is not None and collaborative_weight is not None:
        try:
            total = float(content_weight) + float(collaborative_weight)

            if total > 0:
                st.progress(
                    min(max(float(content_weight) / total, 0), 1),
                    text="Content contribution"
                )
        except (ValueError, TypeError):
            pass


# ============================================================
# COMPONENT INSIGHTS
# ============================================================

def render_component_insights(model):
    content_model, collaborative_model, hybrid_model = get_components(model)

    render_html("""
        <div class="insight-section">
            Inside the recommendation engine.
        </div>

        <div class="insight-caption">
            Inspect the underlying feature and interaction structures.
        </div>
    """)

    components = [
        (
            "CONTENT-BASED ENGINE",
            content_model,
            [
                ("Movie feature matrix", ["movie_features", "tfidf_matrix", "features"]),
                ("Similarity matrix", ["similarity_matrix", "cosine_similarity_matrix"])
            ]
        ),
        (
            "COLLABORATIVE ENGINE",
            collaborative_model,
            [
                ("User-item matrix", ["user_item_matrix", "user_movie_matrix", "ratings_matrix"]),
                ("Similarity matrix", ["similarity_matrix", "user_similarity_matrix"])
            ]
        ),
        (
            "HYBRID ENGINE",
            hybrid_model,
            [
                ("Movie dataset", ["movies"]),
                ("Rating dataset", ["ratings"])
            ]
        )
    ]

    for title, component, attributes in components:
        render_html(f"""
            <div class="insight-panel">
                <div class="insight-panel-title">
                    {html.escape(title)}
                </div>
                <span class="insight-status">
                    {"INITIALIZED" if component is not None else "NOT EXPOSED"}
                </span>
            </div>
        """)

        if component is None:
            continue

        rows = []

        for label, names in attributes:
            value = get_attribute(component, names)

            if value is None:
                display = "Not available"
            else:
                display = get_shape(value)

            rows.append({
                "Component": label,
                "Dimensions": display
            })

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# ARCHITECTURE
# ============================================================

def render_architecture():
    render_html("""
        <div class="insight-section">
            How CineMatch thinks.
        </div>

        <div class="insight-caption">
            High-level workflow of the recommendation pipeline.
        </div>

        <div class="insight-flow">
            <div class="insight-flow-node">MOVIE DATA</div>
            <div class="insight-flow-arrow">→</div>
            <div class="insight-flow-node">FEATURE ENGINE</div>
            <div class="insight-flow-arrow">→</div>
            <div class="insight-flow-node">CONTENT MODEL</div>
        </div>

        <div class="insight-flow">
            <div class="insight-flow-node">USER RATINGS</div>
            <div class="insight-flow-arrow">→</div>
            <div class="insight-flow-node">USER-ITEM MATRIX</div>
            <div class="insight-flow-arrow">→</div>
            <div class="insight-flow-node">COLLABORATIVE MODEL</div>
        </div>

        <div class="insight-flow">
            <div class="insight-flow-node">CONTENT SIGNAL</div>
            <div class="insight-flow-arrow">+</div>
            <div class="insight-flow-node">COLLABORATIVE SIGNAL</div>
            <div class="insight-flow-arrow">→</div>
            <div class="insight-flow-node">HYBRID RANKING</div>
            <div class="insight-flow-arrow">→</div>
            <div class="insight-flow-node">TOP-N MOVIES</div>
        </div>
    """)


# ============================================================
# EVALUATION RESULTS
# ============================================================

def render_evaluation():
    render_html("""
        <div class="insight-section">
            Model evaluation.
        </div>

        <div class="insight-caption">
            Results generated by the evaluation pipeline, if available.
        </div>
    """)

    if not EVALUATION_PATH.exists():
        st.info(
            "No evaluation results file found. "
            "Run the model evaluation pipeline to populate this section."
        )
        return

    try:
        results = pd.read_csv(EVALUATION_PATH)
    except Exception as error:
        st.error("Could not read the evaluation results.")
        st.code(str(error))
        return

    if results.empty:
        st.info("The evaluation results file is empty.")
        return

    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True
    )

    numeric_columns = results.select_dtypes(
        include="number"
    ).columns.tolist()

    if numeric_columns:
        model_column = find_column(
            results,
            ["model", "model_name", "algorithm", "approach"]
        )

        if model_column:
            chart_data = results.set_index(model_column)[numeric_columns]

            st.markdown("#### Evaluation metric comparison")
            st.bar_chart(chart_data)


# ============================================================
# DATA DISTRIBUTION
# ============================================================

def render_data_distribution(stats):
    ratings = stats["ratings"]
    rating_col = stats["rating_col"]

    render_html("""
        <div class="insight-section">
            Rating distribution.
        </div>

        <div class="insight-caption">
            How users have rated movies in the dataset.
        </div>
    """)

    if ratings.empty or rating_col is None:
        st.info("Rating distribution is not available.")
        return

    distribution = (
        ratings[rating_col]
        .value_counts()
        .sort_index()
        .rename_axis("Rating")
        .to_frame("Count")
    )

    st.bar_chart(distribution)

    st.dataframe(
        distribution,
        use_container_width=True
    )


# ============================================================
# MAIN PAGE
# ============================================================

def render_model_insights(model):
    inject_styles()

    render_html("""
        <section class="insight-hero">
            <div class="insight-eyebrow">
                SYSTEM INTELLIGENCE / MODEL LAB
            </div>

            <div class="insight-heading">
                UNDER THE HOOD.
                <span>BEYOND THE SCREEN.</span>
            </div>

            <div class="insight-description">
                Explore the data, algorithms and model architecture
                powering CineMatch. A transparent look at the
                intelligence behind every recommendation.
            </div>
        </section>
    """)

    if model is None:
        st.error("Model is not loaded.")
        return

    stats = get_dataset_stats(model)

    render_overview(stats)

    render_hybrid_weights(model)

    tab1, tab2, tab3, tab4 = st.tabs([
        "MODEL COMPONENTS",
        "ARCHITECTURE",
        "EVALUATION",
        "DATA DISTRIBUTION"
    ])

    with tab1:
        render_component_insights(model)

    with tab2:
        render_architecture()

    with tab3:
        render_evaluation()

    with tab4:
        render_data_distribution(stats)
