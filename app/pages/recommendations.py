
import html
import inspect
import re

import pandas as pd
import streamlit as st


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
    .rec-hero {
        padding:38px;
        border-radius:20px;
        border:1px solid rgba(201,244,90,.16);
        background:
            radial-gradient(circle at 90% 15%,rgba(201,244,90,.17),transparent 35%),
            linear-gradient(120deg,#172019,#0b100c);
        margin-bottom:26px;
    }

    .rec-eyebrow {
        color:#c9f45a;
        font-size:9px;
        font-weight:700;
        letter-spacing:3px;
        margin-bottom:13px;
    }

    .rec-heading {
        color:#f5f7ef;
        font:700 clamp(32px,4vw,48px)/1.1 'Space Grotesk',sans-serif;
        letter-spacing:-2px;
    }

    .rec-heading span { color:#c9f45a; }

    .rec-description {
        color:#a2ad9d;
        font-size:13px;
        line-height:1.8;
        max-width:650px;
        margin-top:12px;
    }

    .rec-section {
        color:#f4f5ef;
        font:700 22px 'Space Grotesk',sans-serif;
        margin:27px 0 5px;
    }

    .rec-caption {
        color:#879282;
        font-size:11px;
        margin-bottom:16px;
    }

    .rec-card {
        background:linear-gradient(145deg,#171d16,#0d110e);
        border:1px solid rgba(255,255,255,.08);
        border-radius:15px;
        padding:17px;
        margin-bottom:10px;
        transition:.25s ease;
    }

    .rec-card:hover {
        transform:translateY(-4px);
        border-color:rgba(201,244,90,.32);
        box-shadow:0 12px 30px rgba(0,0,0,.22);
    }

    .rec-rank {
        color:#c9f45a;
        font:700 10px 'DM Sans',sans-serif;
        letter-spacing:2px;
    }

    .rec-title {
        color:#f4f5ef;
        font:700 17px 'Space Grotesk',sans-serif;
        margin-top:8px;
        line-height:1.35;
    }

    .rec-genres {
        color:#939d8e;
        font-size:10px;
        margin-top:7px;
    }

    .rec-score {
        color:#d9f889;
        font:700 20px 'Space Grotesk',sans-serif;
    }

    .rec-reason {
        color:#a4ae9d;
        font-size:11px;
        line-height:1.7;
        margin-top:10px;
    }

    .rec-chip {
        display:inline-block;
        padding:6px 9px;
        margin:8px 5px 0 0;
        color:#d9f889;
        border:1px solid rgba(201,244,90,.17);
        background:rgba(201,244,90,.06);
        border-radius:30px;
        font-size:9px;
        letter-spacing:1px;
    }

    .rec-empty {
        padding:35px;
        border:1px solid rgba(255,255,255,.08);
        border-radius:15px;
        background:#111611;
        text-align:center;
        color:#a3ad9d;
    }
    </style>
    """, unsafe_allow_html=True)


# ============================================================
# DATA HELPERS
# ============================================================

def get_dataframe(obj, attribute):
    try:
        value = getattr(obj, attribute, None)
        if isinstance(value, pd.DataFrame):
            return value.copy()
    except Exception:
        pass
    return pd.DataFrame()


def find_column(df, candidates):
    lookup = {
        str(column).lower().strip(): column
        for column in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


def safe_text(value):
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value)


def get_user_ids(ratings):
    user_col = find_column(
        ratings,
        ["user_id", "userid", "user"]
    )

    if user_col is None:
        return [], None

    ids = sorted(
        ratings[user_col].dropna().unique().tolist(),
        key=lambda value: str(value)
    )

    return ids, user_col


def get_movie_columns(movies):
    return (
        find_column(movies, ["movie_id", "movieid", "id"]),
        find_column(movies, ["title", "movie_title", "name"]),
        find_column(movies, ["genres", "genre", "category"])
    )


# ============================================================
# USER PROFILE
# ============================================================

def get_user_profile(ratings, user_id, movies):
    user_col = find_column(ratings, ["user_id", "userid", "user"])
    movie_col = find_column(ratings, ["movie_id", "movieid"])
    rating_col = find_column(ratings, ["rating", "score"])

    if not all([user_col, movie_col, rating_col]):
        return pd.DataFrame()

    user_ratings = ratings[
        ratings[user_col] == user_id
    ].copy()

    movie_id_col, title_col, genre_col = get_movie_columns(movies)

    if movie_id_col and not movies.empty:
        user_ratings = user_ratings.merge(
            movies,
            left_on=movie_col,
            right_on=movie_id_col,
            how="left"
        )

    return user_ratings


def render_user_profile(ratings, user_id, movies):
    profile = get_user_profile(ratings, user_id, movies)

    rating_col = find_column(
        profile,
        ["rating", "score"]
    )

    if profile.empty or rating_col is None:
        st.info("No rating history found for this user.")
        return

    total = len(profile)
    average = float(profile[rating_col].mean())
    favourites = int((profile[rating_col] >= 4).sum())

    c1, c2, c3 = st.columns(3)

    metrics = [
        ("RATINGS GIVEN", f"{total:,}"),
        ("AVERAGE RATING", f"{average:.2f}/5"),
        ("HIGHLY RATED", f"{favourites:,}")
    ]

    for col, (label, value) in zip([c1, c2, c3], metrics):
        with col:
            render_html(f"""
                <div class="rec-card">
                    <div class="rec-rank">{label}</div>
                    <div class="rec-score">{value}</div>
                </div>
            """)

    genre_col = find_column(
        profile,
        ["genres", "genre", "category"]
    )

    if genre_col:
        favourites_df = profile[
            profile[rating_col] >= 4
        ]

        genres = []

        for value in favourites_df[genre_col].dropna():
            genres.extend(
                item.strip()
                for item in str(value).replace("|", "·").split("·")
                if item.strip()
            )

        if genres:
            from collections import Counter

            counts = Counter(genres).most_common(5)

            render_html("""
                <div class="rec-section">Your taste profile</div>
                <div class="rec-caption">
                    Most common genres among your highly rated films.
                </div>
            """)

            render_html("".join(
                f'<span class="rec-chip">{html.escape(name)} · {count}</span>'
                for name, count in counts
            ))


# ============================================================
# RECOMMENDATION METHOD ADAPTER
# ============================================================

def call_recommendation_method(method, user_id, top_n):
    """
    Try common argument names without assuming a specific
    implementation signature.
    """
    signature = inspect.signature(method)
    parameters = signature.parameters

    kwargs = {}

    user_names = ["user_id", "user", "uid"]
    count_names = ["top_n", "n_recommendations", "k", "limit", "n"]

    for name in user_names:
        if name in parameters:
            kwargs[name] = user_id
            break

    for name in count_names:
        if name in parameters:
            kwargs[name] = top_n
            break

    if kwargs:
        return method(**kwargs)

    return method(user_id, top_n)


def get_recommendations(model, user_id, top_n):
    candidate_objects = [
        model,
        getattr(model, "hybrid_model", None),
        getattr(model, "hybrid_recommender", None),
        getattr(model, "recommendation_engine", None)
    ]

    method_names = [
        "recommend_for_user",
        "get_recommendations",
        "recommend"
    ]

    errors = []

    for obj in candidate_objects:
        if obj is None:
            continue

        for method_name in method_names:
            method = getattr(obj, method_name, None)

            if not callable(method):
                continue

            try:
                result = call_recommendation_method(
                    method,
                    user_id,
                    top_n
                )

                if result is not None:
                    return result, None

            except Exception as error:
                errors.append(
                    f"{type(obj).__name__}.{method_name}: {error}"
                )

    return None, errors


# ============================================================
# NORMALIZE RESULTS
# ============================================================

def normalize_recommendations(result, movies):
    if result is None:
        return pd.DataFrame()

    if isinstance(result, pd.DataFrame):
        recommendations = result.copy()

    elif isinstance(result, (list, tuple)):
        if len(result) == 0:
            return pd.DataFrame()

        if isinstance(result[0], dict):
            recommendations = pd.DataFrame(result)

        elif isinstance(result[0], (list, tuple)) and len(result[0]) >= 2:
            recommendations = pd.DataFrame(
                result,
                columns=["movie_id", "score"][:len(result[0])]
            )

        else:
            recommendations = pd.DataFrame({
                "movie_id": list(result)
            })

    elif isinstance(result, dict):
        recommendations = pd.DataFrame(result)

    else:
        return pd.DataFrame()

    if recommendations.empty:
        return recommendations

    movie_id_col, title_col, genre_col = get_movie_columns(movies)

    recommendation_id = find_column(
        recommendations,
        ["movie_id", "movieid", "id"]
    )

    if (
        recommendation_id
        and movie_id_col
        and title_col
        and title_col not in recommendations.columns
    ):
        recommendations = recommendations.merge(
            movies,
            left_on=recommendation_id,
            right_on=movie_id_col,
            how="left"
        )

    title_col = find_column(
        recommendations,
        ["title", "movie_title", "name"]
    )

    if title_col is None:
        recommendations["title"] = (
            recommendations[recommendation_id]
            .astype(str)
            if recommendation_id
            else "Recommended movie"
        )
        title_col = "title"

    score_col = find_column(
        recommendations,
        [
            "score",
            "hybrid_score",
            "recommendation_score",
            "predicted_rating",
            "similarity",
            "weighted_score"
        ]
    )

    if score_col is None:
        recommendations["_display_score"] = 0.0
    else:
        recommendations["_display_score"] = pd.to_numeric(
            recommendations[score_col],
            errors="coerce"
        ).fillna(0)

    genre_col = find_column(
        recommendations,
        ["genres", "genre", "category"]
    )

    if genre_col is None:
        recommendations["_display_genres"] = "Genre unavailable"
    else:
        recommendations["_display_genres"] = (
            recommendations[genre_col].fillna("Genre unavailable").astype(str)
        )

    recommendations["_display_title"] = (
        recommendations[title_col].fillna("Unknown movie").astype(str)
    )

    recommendations = recommendations.drop_duplicates(
        subset=["_display_title"]
    )

    return recommendations


# ============================================================
# RECOMMENDATION CARDS
# ============================================================

def render_recommendation_card(movie, index):
    title = html.escape(str(movie["_display_title"]))
    genres = html.escape(str(movie["_display_genres"]))

    score = float(movie["_display_score"])

    render_html(f"""
        <div class="rec-card">
            <div class="rec-rank">MATCH NO. {index:02d}</div>

            <div class="rec-title">{title}</div>

            <div class="rec-genres">{genres}</div>

            <div class="rec-score">
                {score:.3f}
            </div>

            <div class="rec-reason">
                Selected by your recommendation model.
            </div>
        </div>
    """)


# ============================================================
# MAIN PAGE
# ============================================================

def render_recommendations(model):
    inject_styles()

    render_html("""
        <section class="rec-hero">
            <div class="rec-eyebrow">
                PERSONALIZED DISCOVERY / AI ENGINE
            </div>

            <div class="rec-heading">
                YOUR TASTE.
                <span>YOUR CINEMA.</span>
            </div>

            <div class="rec-description">
                Discover movies selected for your preferences.
                Your recommendation engine combines available
                movie and user-rating information to find
                your next watch.
            </div>
        </section>
    """)

    movies = get_dataframe(model, "movies")
    ratings = get_dataframe(model, "ratings")

    if movies.empty:
        st.error("Movie data is unavailable.")
        return

    if ratings.empty:
        st.error("Rating data is unavailable.")
        return

    user_ids, user_col = get_user_ids(ratings)

    if not user_ids:
        st.error("No user IDs were found in the ratings dataset.")
        return

    render_html("""
        <div class="rec-section">Choose your profile.</div>
        <div class="rec-caption">
            Select a dataset user to generate personalized recommendations.
        </div>
    """)

    default_user = st.session_state.get("selected_user", user_ids[0])

    if default_user not in user_ids:
        default_user = user_ids[0]

    selected_user = st.selectbox(
        "SELECT USER PROFILE",
        options=user_ids,
        index=user_ids.index(default_user),
        format_func=lambda value: f"User #{value}",
        key="recommendation_user"
    )

    st.session_state.selected_user = selected_user

    render_user_profile(
        ratings,
        selected_user,
        movies
    )

    top_n = st.slider(
        "NUMBER OF RECOMMENDATIONS",
        min_value=5,
        max_value=30,
        value=10,
        step=5
    )

    if st.button(
        "✳ GENERATE MY RECOMMENDATIONS",
        use_container_width=True,
        type="primary",
        key="generate_recommendations"
    ):
        with st.spinner("Analyzing your movie taste..."):
            result, errors = get_recommendations(
                model,
                selected_user,
                top_n
            )

            if result is None:
                st.session_state.recommendation_results = None
                st.session_state.recommendation_error = errors
            else:
                normalized = normalize_recommendations(result, movies)

                st.session_state.recommendation_results = normalized
                st.session_state.recommendation_error = None

                st.session_state.recommendation_user_result = selected_user

    if "recommendation_results" in st.session_state:
        result = st.session_state.recommendation_results

        if result is None:
            st.error(
                "The model did not return recommendations. "
                "Check the available recommendation method."
            )

            errors = st.session_state.get("recommendation_error")

            if errors:
                with st.expander("Technical details"):
                    for error in errors:
                        st.write(error)

        elif result.empty:
            st.info("No recommendations were returned for this profile.")

        else:
            render_html("""
                <div class="rec-section">
                    Your next watchlist.
                </div>
                <div class="rec-caption">
                    Ranked results from the recommendation engine.
                </div>
            """)

            st.success(
                f"{len(result)} recommendations generated for "
                f"User #{st.session_state.recommendation_user_result}."
            )

            for index, (_, movie) in enumerate(
                result.head(top_n).iterrows(),
                start=1
            ):
                render_recommendation_card(movie, index)
