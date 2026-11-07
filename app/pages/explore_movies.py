
import html
import re
import textwrap

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
# PAGE STYLES
# ============================================================

def inject_explore_styles():
    st.markdown("""
    <style>
    .explore-hero {
        position:relative;
        overflow:hidden;
        padding:38px 38px 35px;
        border-radius:20px;
        margin-bottom:28px;
        background:
            radial-gradient(ellipse at 90% 20%,rgba(101,229,209,.13),transparent 35%),
            radial-gradient(ellipse at 10% 100%,rgba(201,244,90,.12),transparent 40%),
            linear-gradient(120deg,#172019,#0c100d);
        border:1px solid rgba(201,244,90,.14);
    }

    .explore-eyebrow {
        color:#c9f45a;
        font-size:9px;
        font-weight:700;
        letter-spacing:3px;
        margin-bottom:13px;
    }

    .explore-heading {
        color:#f5f7ef;
        font:700 clamp(32px,4vw,48px)/1.1 'Space Grotesk',sans-serif;
        letter-spacing:-2px;
        margin:0;
    }

    .explore-heading span {
        color:#c9f45a;
    }

    .explore-description {
        color:#a1ab9d;
        font-size:13px;
        line-height:1.8;
        max-width:620px;
        margin-top:13px;
    }

    .explore-counter {
        display:inline-flex;
        gap:8px;
        align-items:center;
        padding:9px 13px;
        border-radius:30px;
        background:rgba(201,244,90,.08);
        border:1px solid rgba(201,244,90,.17);
        color:#d8f68e;
        font-size:10px;
        letter-spacing:1px;
        margin-top:16px;
    }

    .explore-toolbar {
        display:flex;
        justify-content:space-between;
        align-items:center;
        margin:25px 0 15px;
    }

    .explore-section-title {
        color:#f4f5ef;
        font:700 22px 'Space Grotesk',sans-serif;
    }

    .explore-section-subtitle {
        color:#858f81;
        font-size:11px;
        margin-top:5px;
    }

    .explore-movie-card {
        position:relative;
        min-height:285px;
        overflow:hidden;
        border:1px solid rgba(255,255,255,.08);
        border-radius:15px;
        background:#121712;
        transition:transform .25s ease,border-color .25s ease,box-shadow .25s ease;
        margin-bottom:9px;
    }

    .explore-movie-card:hover {
        transform:translateY(-6px);
        border-color:rgba(201,244,90,.4);
        box-shadow:0 18px 38px rgba(0,0,0,.35);
    }

    .explore-poster {
        height:175px;
        position:relative;
        display:flex;
        align-items:center;
        justify-content:center;
        overflow:hidden;
    }

    .explore-poster::after {
        content:"";
        position:absolute;
        inset:0;
        background:linear-gradient(180deg,transparent 45%,rgba(0,0,0,.65));
    }

    .explore-poster-symbol {
        font:700 67px 'Space Grotesk',sans-serif;
        color:rgba(255,255,255,.23);
        z-index:1;
        text-shadow:0 0 35px rgba(255,255,255,.13);
    }

    .explore-poster-number {
        position:absolute;
        z-index:2;
        top:12px;
        left:13px;
        color:rgba(255,255,255,.8);
        font:700 9px 'DM Sans',sans-serif;
        letter-spacing:2px;
    }

    .explore-poster-rating {
        position:absolute;
        z-index:2;
        top:10px;
        right:10px;
        background:rgba(0,0,0,.55);
        border:1px solid rgba(255,255,255,.12);
        color:#d9f889;
        border-radius:7px;
        padding:5px 8px;
        font-size:10px;
        font-weight:700;
    }

    .explore-card-body {
        padding:14px 14px 13px;
    }

    .explore-card-title {
        color:#f4f5ef;
        font:600 13px 'Space Grotesk',sans-serif;
        line-height:1.4;
        min-height:36px;
    }

    .explore-card-genre {
        color:#929d8d;
        font-size:9px;
        line-height:1.6;
        margin-top:7px;
        min-height:28px;
    }

    .explore-card-footer {
        display:flex;
        justify-content:space-between;
        align-items:center;
        color:#778171;
        font-size:9px;
        margin-top:10px;
        padding-top:9px;
        border-top:1px solid rgba(255,255,255,.07);
    }

    .explore-card-footer strong {
        color:#c9f45a;
    }

    .explore-empty {
        text-align:center;
        padding:55px 20px;
        background:#111611;
        border:1px solid rgba(255,255,255,.07);
        border-radius:16px;
    }

    .explore-empty-icon {
        color:#c9f45a;
        font-size:35px;
        margin-bottom:12px;
    }

    .explore-empty-title {
        color:#f4f5ef;
        font:700 20px 'Space Grotesk',sans-serif;
    }

    .explore-empty-description {
        color:#8b9586;
        font-size:12px;
        margin-top:8px;
    }

    .explore-detail {
        padding:22px;
        background:linear-gradient(140deg,#171d16,#0d110e);
        border:1px solid rgba(201,244,90,.17);
        border-radius:15px;
        margin:15px 0;
    }

    .explore-detail-label {
        color:#c9f45a;
        font-size:9px;
        font-weight:700;
        letter-spacing:2px;
    }

    .explore-detail-title {
        color:#f4f5ef;
        font:700 24px 'Space Grotesk',sans-serif;
        margin-top:10px;
    }

    .explore-detail-meta {
        color:#9aa494;
        font-size:12px;
        margin-top:8px;
        line-height:1.8;
    }

    div[data-testid="stMultiSelect"] label,
    div[data-testid="stSelectbox"] label,
    div[data-testid="stSlider"] label,
    div[data-testid="stTextInput"] label {
        color:#b5bfb0 !important;
    }

    @media(max-width:700px) {
        .explore-hero {
            padding:25px 22px;
        }

        .explore-heading {
            letter-spacing:-1px;
        }
    }
    </style>
    """, unsafe_allow_html=True)


# ============================================================
# DATA HELPERS
# ============================================================

def get_dataframe(model, attribute):
    try:
        data = getattr(model, attribute, None)
        if isinstance(data, pd.DataFrame):
            return data.copy()
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


def extract_year(title):
    match = re.search(r"\((\d{4})\)\s*$", str(title))
    return int(match.group(1)) if match else None


def clean_title(title):
    return re.sub(r"\s*\(\d{4}\)\s*$", "", str(title)).strip()


def format_genres(value):
    if isinstance(value, (list, tuple, set)):
        return " · ".join(str(x) for x in value)

    if pd.isna(value):
        return "Genre unavailable"

    return str(value).replace("|", " · ")


def get_movie_symbol(title):
    words = re.findall(r"[A-Za-z0-9]", str(title))
    return "".join(words[:2]).upper() or "CM"


# ============================================================
# MOVIE DATA PREPARATION
# ============================================================

def prepare_movies(model):
    movies = get_dataframe(model, "movies")
    ratings = get_dataframe(model, "ratings")

    if movies.empty:
        return pd.DataFrame()

    movie_id_col = find_column(
        movies,
        ["movie_id", "movieid", "id"]
    )

    title_col = find_column(
        movies,
        ["title", "movie_title", "name"]
    )

    genre_col = find_column(
        movies,
        ["genres", "genre", "category"]
    )

    if title_col is None:
        return pd.DataFrame()

    movies = movies.copy()

    movies["_display_title"] = movies[title_col].astype(str)
    movies["_year"] = movies["_display_title"].apply(extract_year)
    movies["_clean_title"] = movies["_display_title"].apply(clean_title)

    if genre_col:
        movies["_genres"] = movies[genre_col].apply(format_genres)
    else:
        movies["_genres"] = "Genre unavailable"

    movies["_rating"] = 0.0
    movies["_rating_count"] = 0

    rating_movie_col = find_column(
        ratings,
        ["movie_id", "movieid"]
    )

    rating_col = find_column(
        ratings,
        ["rating", "score"]
    )

    if (
        not ratings.empty
        and movie_id_col
        and rating_movie_col
        and rating_col
    ):
        rating_stats = (
            ratings.groupby(rating_movie_col)[rating_col]
            .agg(["mean", "count"])
            .reset_index()
        )

        rating_stats.columns = [
            rating_movie_col,
            "_rating",
            "_rating_count"
        ]

        movies = movies.merge(
            rating_stats,
            left_on=movie_id_col,
            right_on=rating_movie_col,
            how="left",
            suffixes=("", "_calculated")
        )

        if "_rating_calculated" in movies.columns:
            movies["_rating"] = movies["_rating_calculated"].fillna(0)
            movies.drop(columns=["_rating_calculated"], inplace=True)

        if "_rating_count_calculated" in movies.columns:
            movies["_rating_count"] = (
                movies["_rating_count_calculated"].fillna(0)
            )
            movies.drop(
                columns=["_rating_count_calculated"],
                inplace=True
            )

    movies["_rating"] = pd.to_numeric(
        movies["_rating"], errors="coerce"
    ).fillna(0)

    movies["_rating_count"] = pd.to_numeric(
        movies["_rating_count"], errors="coerce"
    ).fillna(0)

    return movies


# ============================================================
# GENRE FILTERING
# ============================================================

def get_all_genres(movies):
    genres = set()

    if movies.empty:
        return []

    for value in movies["_genres"]:
        if not isinstance(value, str):
            continue

        for genre in value.split(" · "):
            genre = genre.strip()

            if genre and genre.lower() not in {
                "unknown", "genre unavailable"
            }:
                genres.add(genre)

    return sorted(genres)


# ============================================================
# FILTER ENGINE
# ============================================================

def apply_filters(
    movies,
    search_query,
    selected_genres,
    minimum_rating,
    minimum_votes,
    sort_option
):
    filtered = movies.copy()

    if search_query.strip():
        query = search_query.strip().lower()

        filtered = filtered[
            filtered["_display_title"].str.lower().str.contains(
                query,
                na=False,
                regex=False
            )
        ]

    if selected_genres:
        pattern = "|".join(
            re.escape(genre)
            for genre in selected_genres
        )

        filtered = filtered[
            filtered["_genres"].str.contains(
                pattern,
                case=False,
                regex=True,
                na=False
            )
        ]

    filtered = filtered[
        filtered["_rating"] >= minimum_rating
    ]

    filtered = filtered[
        filtered["_rating_count"] >= minimum_votes
    ]

    sort_mapping = {
        "Most Popular": ["_rating_count", "_rating"],
        "Highest Rated": ["_rating", "_rating_count"],
        "Title A-Z": ["_clean_title"],
        "Newest First": ["_year", "_rating"],
        "Oldest First": ["_year", "_rating"]
    }

    columns = sort_mapping.get(sort_option, ["_rating_count"])

    ascending = sort_option in ["Title A-Z", "Oldest First"]

    filtered = filtered.sort_values(
        by=columns,
        ascending=ascending,
        na_position="last"
    )

    return filtered


# ============================================================
# HERO
# ============================================================

def render_hero(total_movies):
    render_html(f"""
        <section class="explore-hero">
            <div class="explore-eyebrow">
                CINEMATIC UNIVERSE / EXPLORE
            </div>

            <h1 class="explore-heading">
                THE WORLD OF
                <span>ENDLESS STORIES.</span>
            </h1>

            <div class="explore-description">
                From timeless classics to hidden gems.
                Search, filter and discover your next favourite
                film from the CineMatch collection.
            </div>

            <div class="explore-counter">
                ◈ &nbsp; {total_movies:,} MOVIES IN YOUR UNIVERSE
            </div>
        </section>
    """)


# ============================================================
# MOVIE CARD
# ============================================================

def poster_palette(index):
    palettes = [
        "linear-gradient(145deg,#29483e,#101a19 75%)",
        "linear-gradient(145deg,#604034,#211411 75%)",
        "linear-gradient(145deg,#3b3566,#17152e 75%)",
        "linear-gradient(145deg,#69542c,#282010 75%)",
        "linear-gradient(145deg,#28546a,#10232e 75%)",
        "linear-gradient(145deg,#673b50,#291522 75%)",
        "linear-gradient(145deg,#4b592f,#1c2616 75%)",
        "linear-gradient(145deg,#45464f,#1b1c23 75%)"
    ]

    return palettes[index % len(palettes)]


def render_movie_card(movie, index, id_column):
    title = str(movie["_display_title"])
    safe_title = html.escape(title)

    genre = html.escape(str(movie["_genres"]))
    symbol = html.escape(get_movie_symbol(title))

    rating = float(movie["_rating"])
    votes = int(movie["_rating_count"])

    year = movie["_year"]
    year_text = str(int(year)) if pd.notna(year) else "YEAR N/A"

    movie_id = (
        movie.get(id_column)
        if id_column
        else None
    )

    render_html(f"""
        <div class="explore-movie-card">
            <div class="explore-poster"
                 style="background:{poster_palette(index)}">

                <div class="explore-poster-number">
                    CINEMATCH / {index + 1:03d}
                </div>

                <div class="explore-poster-rating">
                    ★ {rating:.1f}
                </div>

                <div class="explore-poster-symbol">
                    {symbol}
                </div>
            </div>

            <div class="explore-card-body">
                <div class="explore-card-title">
                    {safe_title[:85]}
                </div>

                <div class="explore-card-genre">
                    {genre[:75]}
                </div>

                <div class="explore-card-footer">
                    <span>{year_text}</span>
                    <strong>{votes:,} RATINGS</strong>
                </div>
            </div>
        </div>
    """)

    if st.button(
        "VIEW DETAILS ↗",
        key=f"explore_view_{index}_{movie_id}",
        use_container_width=True
    ):
        st.session_state.selected_movie = {
            "id": movie_id,
            "title": title,
            "genres": str(movie["_genres"]),
            "rating": rating,
            "rating_count": votes,
            "year": year_text
        }

        st.rerun()


# ============================================================
# SELECTED MOVIE DETAILS
# ============================================================

def render_selected_movie():
    selected = st.session_state.get("selected_movie")

    if not isinstance(selected, dict):
        return

    if not selected.get("title"):
        return

    render_html(f"""
        <div class="explore-detail">
            <div class="explore-detail-label">
                SELECTED FILM / MOVIE PROFILE
            </div>

            <div class="explore-detail-title">
                {html.escape(str(selected.get("title", "")))}
            </div>

            <div class="explore-detail-meta">
                Genre: {html.escape(str(selected.get("genres", "N/A")))}<br>
                Year: {html.escape(str(selected.get("year", "N/A")))}<br>
                Community rating: {float(selected.get("rating", 0)):.2f}/5<br>
                Rating count: {int(selected.get("rating_count", 0)):,}
            </div>
        </div>
    """)

    if st.button(
        "CLEAR SELECTION",
        key="clear_selected_movie"
    ):
        st.session_state.selected_movie = None
        st.rerun()


# ============================================================
# FILTER SIDEBAR AREA
# ============================================================

def render_filters(movies):
    all_genres = get_all_genres(movies)

    col1, col2 = st.columns([1.5, 1])

    with col1:
        search_query = st.text_input(
            "SEARCH YOUR NEXT MOVIE",
            placeholder="Search by movie title...",
            key="explore_search"
        )

    with col2:
        sort_option = st.selectbox(
            "SORT COLLECTION",
            [
                "Most Popular",
                "Highest Rated",
                "Title A-Z",
                "Newest First",
                "Oldest First"
            ],
            key="explore_sort"
        )

    filter_col1, filter_col2, filter_col3 = st.columns([1.5, 1, 1])

    with filter_col1:
        selected_genres = st.multiselect(
            "FILTER BY GENRE",
            options=all_genres,
            placeholder="Choose genres..."
        )

    with filter_col2:
        minimum_rating = st.slider(
            "MINIMUM RATING",
            min_value=0.0,
            max_value=5.0,
            value=0.0,
            step=0.5
        )

    with filter_col3:
        maximum_votes = int(movies["_rating_count"].max()) if not movies.empty else 0

        vote_limit = min(maximum_votes, 100)

        minimum_votes = st.select_slider(
            "MINIMUM VOTES",
            options=[0, 5, 10, 25, 50, 100, 250, 500],
            value=0
        )

        minimum_votes = min(minimum_votes, maximum_votes)

    return (
        search_query,
        selected_genres,
        minimum_rating,
        minimum_votes,
        sort_option
    )


# ============================================================
# COLLECTION GRID
# ============================================================

def render_collection(filtered, id_column):
    render_html(f"""
        <div class="explore-toolbar">
            <div>
                <div class="explore-section-title">
                    Discover the collection.
                </div>
                <div class="explore-section-subtitle">
                    {len(filtered):,} matching movies found
                </div>
            </div>
        </div>
    """)

    if filtered.empty:
        render_html("""
            <div class="explore-empty">
                <div class="explore-empty-icon">⌕</div>
                <div class="explore-empty-title">
                    No movies found.
                </div>
                <div class="explore-empty-description">
                    Try changing your search or adjusting the filters.
                </div>
            </div>
        """)
        return

    page_size = 12
    total_pages = max(1, (len(filtered) + page_size - 1) // page_size)

    if "explore_page" not in st.session_state:
        st.session_state.explore_page = 1

    if st.session_state.explore_page > total_pages:
        st.session_state.explore_page = 1

    page = st.session_state.explore_page

    start = (page - 1) * page_size
    end = start + page_size

    visible_movies = filtered.iloc[start:end]

    columns = st.columns(4)

    for index, (_, movie) in enumerate(visible_movies.iterrows()):
        with columns[index % 4]:
            render_movie_card(
                movie,
                start + index,
                id_column
            )

    st.markdown("---")

    previous_col, page_col, next_col = st.columns([1, 2, 1])

    with previous_col:
        if st.button(
            "← PREVIOUS",
            disabled=page <= 1,
            use_container_width=True,
            key="explore_previous"
        ):
            st.session_state.explore_page -= 1
            st.rerun()

    with page_col:
        st.markdown(
            f"<div style='text-align:center;color:#a4ae9d;padding:10px'>"
            f"PAGE {page} OF {total_pages}</div>",
            unsafe_allow_html=True
        )

    with next_col:
        if st.button(
            "NEXT →",
            disabled=page >= total_pages,
            use_container_width=True,
            key="explore_next"
        ):
            st.session_state.explore_page += 1
            st.rerun()


# ============================================================
# MAIN EXPLORE PAGE
# ============================================================

def render_explore_movies(model):
    inject_explore_styles()

    movies = prepare_movies(model)

    if movies.empty:
        st.error(
            "Movie dataset is unavailable or its title column "
            "could not be identified."
        )
        return

    render_hero(len(movies))

    (
        search_query,
        selected_genres,
        minimum_rating,
        minimum_votes,
        sort_option
    ) = render_filters(movies)

    filtered = apply_filters(
        movies,
        search_query,
        selected_genres,
        minimum_rating,
        minimum_votes,
        sort_option
    )

    # Reset pagination whenever filters change
    filter_signature = (
        search_query,
        tuple(selected_genres),
        minimum_rating,
        minimum_votes,
        sort_option
    )

    if st.session_state.get("explore_filter_signature") != filter_signature:
        st.session_state.explore_filter_signature = filter_signature
        st.session_state.explore_page = 1

    id_column = find_column(
        movies,
        ["movie_id", "movieid", "id"]
    )

    render_collection(filtered, id_column)

    render_selected_movie()
