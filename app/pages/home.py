
import html
import re
import hashlib

import pandas as pd
import streamlit as st


# ==================================================
# HTML RENDERER
# ==================================================

def render_html(content):
    clean_html = "\n".join(
        line.lstrip()
        for line in content.strip().splitlines()
    )
    st.markdown(clean_html, unsafe_allow_html=True)


# ==================================================
# HOME STYLES
# ==================================================

def inject_home_styles():
    st.markdown("""
    <style>
    .home-hero {
        position:relative;
        min-height:430px;
        overflow:hidden;
        display:flex;
        align-items:center;
        padding:55px 50px;
        margin:5px 0 30px;
        border-radius:23px;
        border:1px solid rgba(201,244,90,.16);
        background:
            radial-gradient(ellipse at 78% 40%,rgba(201,244,90,.20),transparent 32%),
            radial-gradient(ellipse at 95% 90%,rgba(67,216,197,.12),transparent 40%),
            linear-gradient(110deg,#171e16,#101510 48%,#090d0a);
        isolation:isolate;
    }

    .home-hero::before {
        content:"";
        position:absolute;
        width:430px;
        height:430px;
        right:3%;
        top:0;
        border-radius:50%;
        border:1px solid rgba(201,244,90,.18);
        box-shadow:
            0 0 0 35px rgba(201,244,90,.025),
            0 0 0 75px rgba(201,244,90,.018),
            inset 0 0 80px rgba(201,244,90,.06);
        animation:cmOrbit 8s ease-in-out infinite alternate;
        z-index:-1;
    }

    .home-hero::after {
        content:"C";
        position:absolute;
        right:9%;
        top:2%;
        color:rgba(201,244,90,.045);
        font:700 360px/1 'Space Grotesk',sans-serif;
        transform:rotate(-12deg);
        z-index:-1;
    }

    .hero-content {
        max-width:680px;
        position:relative;
        z-index:2;
        animation:cmHeroEntrance .9s ease both;
    }

    .hero-kicker {
        display:flex;
        align-items:center;
        gap:9px;
        color:#d5fa75;
        font-size:10px;
        font-weight:700;
        letter-spacing:3px;
        margin-bottom:23px;
    }

    .hero-kicker-dot {
        width:7px;
        height:7px;
        background:#c9f45a;
        border-radius:50%;
        box-shadow:0 0 14px #c9f45a;
    }

    .hero-title {
        color:#f5f7ef;
        font:700 clamp(43px,6vw,76px)/.99 'Space Grotesk',sans-serif;
        letter-spacing:-4px;
        margin:0 0 23px;
    }

    .hero-title span {
        display:block;
        color:#c9f45a;
        text-shadow:0 0 35px rgba(201,244,90,.17);
    }

    .hero-description {
        color:#a6afa1;
        font-size:14px;
        line-height:1.9;
        max-width:480px;
        margin-bottom:26px;
    }

    .hero-bottom-line {
        display:flex;
        align-items:center;
        flex-wrap:wrap;
        gap:13px;
        color:#7e897a;
        font-size:10px;
        letter-spacing:1.4px;
        font-weight:600;
    }

    .hero-bottom-line strong { color:#e2e9dc; }

    .hero-divider {
        width:28px;
        height:1px;
        background:#64715a;
    }

    .home-section { margin:35px 0 18px; }

    .home-section-kicker {
        color:#c9f45a;
        font-size:9px;
        letter-spacing:2.5px;
        font-weight:700;
        margin-bottom:8px;
    }

    .home-section-title {
        color:#f4f5ef;
        font:700 25px 'Space Grotesk',sans-serif;
        letter-spacing:-.8px;
    }

    .home-section-caption {
        color:#818a7d;
        font-size:11px;
        margin-top:5px;
    }

    .home-stat {
        min-height:110px;
        padding:19px;
        border-radius:15px;
        background:linear-gradient(140deg,#171d16,#101310);
        border:1px solid rgba(255,255,255,.07);
        transition:.25s ease;
    }

    .home-stat:hover {
        transform:translateY(-4px);
        border-color:rgba(201,244,90,.3);
        box-shadow:0 12px 35px rgba(0,0,0,.22);
    }

    .home-stat-label {
        color:#899285;
        font-size:9px;
        font-weight:700;
        letter-spacing:1.5px;
    }

    .home-stat-value {
        color:#f5f7ef;
        font:700 29px 'Space Grotesk',sans-serif;
        margin-top:10px;
    }

    .home-stat-foot {
        color:#75806f;
        font-size:8px;
        letter-spacing:1.3px;
        margin-top:4px;
    }

    .home-poster {
        position:relative;
        height:225px;
        overflow:hidden;
        border-radius:13px;
        display:flex;
        align-items:center;
        justify-content:center;
        isolation:isolate;
        transition:transform .3s ease,box-shadow .3s ease;
        border:1px solid rgba(255,255,255,.09);
    }

    .home-poster::before {
        content:"";
        position:absolute;
        inset:0;
        background:
            radial-gradient(circle at 50% 30%,rgba(255,255,255,.17),transparent 35%),
            linear-gradient(180deg,transparent 25%,rgba(0,0,0,.88));
        z-index:-1;
    }

    .home-poster:hover {
        transform:translateY(-7px) scale(1.015);
        box-shadow:0 18px 35px rgba(0,0,0,.35);
    }

    .poster-symbol {
        font:700 74px 'Space Grotesk',sans-serif;
        color:rgba(255,255,255,.23);
    }

    .poster-number {
        position:absolute;
        top:12px;
        left:13px;
        color:white;
        font:700 9px 'DM Sans',sans-serif;
        letter-spacing:2px;
    }

    .poster-bottom {
        position:absolute;
        left:15px;
        right:12px;
        bottom:15px;
    }

    .poster-tag {
        color:#d9ff7d;
        font-size:8px;
        font-weight:700;
        letter-spacing:2px;
        margin-bottom:6px;
    }

    .poster-title {
        color:white;
        font:700 15px 'Space Grotesk',sans-serif;
        line-height:1.25;
    }

    .poster-meta {
        color:rgba(255,255,255,.66);
        font-size:9px;
        margin-top:5px;
    }

    .home-feature-strip {
        padding:23px 25px;
        background:linear-gradient(100deg,rgba(201,244,90,.10),rgba(101,229,209,.035));
        border:1px solid rgba(201,244,90,.14);
        border-radius:15px;
        display:flex;
        align-items:center;
        justify-content:space-between;
        gap:20px;
        margin:34px 0;
    }

    .feature-strip-label {
        color:#c9f45a;
        font-size:9px;
        font-weight:700;
        letter-spacing:2px;
        margin-bottom:8px;
    }

    .feature-strip-title {
        color:#f4f5ef;
        font:700 19px 'Space Grotesk',sans-serif;
    }

    .feature-strip-desc {
        color:#9aa494;
        font-size:11px;
        margin-top:6px;
    }

    @keyframes cmHeroEntrance {
        from { opacity:0; transform:translateY(25px); }
        to { opacity:1; transform:translateY(0); }
    }

    @keyframes cmOrbit {
        from { transform:rotate(0deg) scale(.96); }
        to { transform:rotate(15deg) scale(1.04); }
    }

    @media(max-width:700px) {
        .home-hero { min-height:360px; padding:30px 25px; }
        .hero-title { letter-spacing:-2px; }
        .home-poster { height:190px; }
    }

    @media(prefers-reduced-motion:reduce) {
        .home-hero::before,.hero-content { animation:none!important; }
    }
    </style>
    """, unsafe_allow_html=True)


# ==================================================
# DATA HELPERS
# ==================================================

def get_dataframe(model, attribute):
    try:
        data = getattr(model, attribute, None)
        if isinstance(data, pd.DataFrame):
            return data.copy()
    except Exception:
        pass

    return pd.DataFrame()


def find_column(df, names):
    lookup = {
        str(column).lower().strip(): column
        for column in df.columns
    }

    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]

    return None


def get_home_data(model):
    movies = get_dataframe(model, "movies")
    ratings = get_dataframe(model, "ratings")

    movie_id = find_column(movies, ["movie_id", "movieid", "id"])
    title = find_column(movies, ["title", "movie_title", "name"])
    genres = find_column(movies, ["genres", "genre", "category"])

    user_id = find_column(ratings, ["user_id", "userid"])
    rating_movie = find_column(ratings, ["movie_id", "movieid"])
    rating = find_column(ratings, ["rating", "score"])

    total_users = (
        ratings[user_id].nunique()
        if user_id and not ratings.empty else 0
    )

    average = (
        float(ratings[rating].mean())
        if rating and not ratings.empty else 0
    )

    trending = pd.DataFrame()

    if (
        not movies.empty and not ratings.empty
        and movie_id and title and rating_movie and rating
    ):
        stats = (
            ratings.groupby(rating_movie)
            .agg(
                average_rating=(rating, "mean"),
                rating_count=(rating, "count")
            )
            .reset_index()
        )

        stats = stats[stats["rating_count"] >= 3]

        trending = stats.merge(
            movies,
            left_on=rating_movie,
            right_on=movie_id,
            how="inner"
        )

        trending = trending.sort_values(
            ["rating_count", "average_rating"],
            ascending=[False, False]
        ).head(8)

    return {
        "movies": movies,
        "ratings": ratings,
        "trending": trending,
        "total_movies": len(movies),
        "total_ratings": len(ratings),
        "total_users": total_users,
        "average_rating": average,
        "title_col": title,
        "genres_col": genres,
        "movie_id_col": movie_id
    }


def safe_number(value):
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "0"


# ==================================================
# HERO
# ==================================================

def render_hero(total_movies):
    render_html(f"""
        <section class="home-hero">
            <div class="hero-content">
                <div class="hero-kicker">
                    <span class="hero-kicker-dot"></span>
                    THE NEXT GENERATION OF MOVIE DISCOVERY
                </div>

                <h1 class="hero-title">
                    FIND YOUR
                    <span>NEXT OBSESSION.</span>
                </h1>

                <p class="hero-description">
                    Forget endless scrolling. Discover films that
                    match your taste, explore hidden gems and let
                    intelligent recommendations lead you to your
                    next unforgettable watch.
                </p>

                <div class="hero-bottom-line">
                    <strong>{safe_number(total_movies)}+ MOVIES</strong>
                    <span class="hero-divider"></span>
                    <span>AI-POWERED DISCOVERY</span>
                    <span class="hero-divider"></span>
                    <span>BUILT FOR CINEPHILES</span>
                </div>
            </div>
        </section>
    """)

    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            "✳   EXPLORE MOVIES",
            key="home_explore_button",
            use_container_width=True
        ):
            st.session_state.current_page = "Explore Movies"
            st.rerun()

    with col2:
        if st.button(
            "◈   GET MY RECOMMENDATIONS",
            key="home_recommend_button",
            use_container_width=True
        ):
            st.session_state.current_page = "Recommendations"
            st.rerun()


# ==================================================
# STATISTICS
# ==================================================

def render_statistics(data):
    render_html("""
        <div class="home-section">
            <div class="home-section-kicker">THE CINEMATCH UNIVERSE</div>
            <div class="home-section-title">Your discovery playground.</div>
            <div class="home-section-caption">
                A quick look at the data powering your experience.
            </div>
        </div>
    """)

    stats = [
        ("MOVIES IN LIBRARY", safe_number(data["total_movies"]), "FILM COLLECTION"),
        ("COMMUNITY RATINGS", safe_number(data["total_ratings"]), "DATA POINTS"),
        ("ACTIVE PROFILES", safe_number(data["total_users"]), "USER PROFILES"),
        (
            "AVERAGE RATING",
            f'{data["average_rating"]:.2f}/5' if data["average_rating"] else "N/A",
            "COMMUNITY SCORE"
        )
    ]

    columns = st.columns(4)

    for column, (label, value, footer) in zip(columns, stats):
        with column:
            render_html(f"""
                <div class="home-stat">
                    <div class="home-stat-label">{html.escape(label)}</div>
                    <div class="home-stat-value">{html.escape(value)}</div>
                    <div class="home-stat-foot">{html.escape(footer)}</div>
                </div>
            """)


# ==================================================
# POSTER DESIGN
# ==================================================

def poster_background(title):
    palettes = [
        "linear-gradient(145deg,#273b34,#101b1a 65%,#080c0d)",
        "linear-gradient(145deg,#573d35,#211716 65%,#0d0b0c)",
        "linear-gradient(145deg,#3c355f,#17172e 65%,#090a13)",
        "linear-gradient(145deg,#594729,#292010 65%,#0c0b08)",
        "linear-gradient(145deg,#274b5a,#10252f 65%,#080d12)",
        "linear-gradient(145deg,#633c4e,#291520 65%,#10090d)",
        "linear-gradient(145deg,#48532c,#202817 65%,#0a0d09)",
        "linear-gradient(145deg,#42424a,#1b1c23 65%,#090a0d)"
    ]

    index = int(
        hashlib.md5(str(title).encode("utf-8")).hexdigest(),
        16
    ) % len(palettes)

    return palettes[index]


def render_movie_poster(title, genre, rating, index, total_ratings):
    safe_title = html.escape(str(title))
    safe_genre = html.escape(str(genre or "CINEMA"))

    background = poster_background(title)

    words = re.findall(r"[A-Za-z0-9]", str(title))
    symbol = "".join(words[:2]).upper() or "CM"

    render_html(f"""
        <div class="home-poster" style="background:{background}">
            <div class="poster-number">NO. {index:02d}</div>
            <div class="poster-symbol">{html.escape(symbol)}</div>

            <div class="poster-bottom">
                <div class="poster-tag">{safe_genre[:45].upper()}</div>
                <div class="poster-title">{safe_title[:65]}</div>
                <div class="poster-meta">
                    ★ {rating:.1f}/5 · {safe_number(total_ratings)} RATINGS
                </div>
            </div>
        </div>
    """)


# ==================================================
# TRENDING MOVIES
# ==================================================

def render_trending(data):
    trending = data["trending"]

    render_html("""
        <div class="home-section">
            <div class="home-section-kicker">COMMUNITY FAVOURITES</div>
            <div class="home-section-title">Trending in CineMatch.</div>
            <div class="home-section-caption">
                Films attracting the most ratings in our dataset.
            </div>
        </div>
    """)

    if trending.empty:
        st.info("Trending movie data is not available yet.")
        return

    title_col = data["title_col"]
    genres_col = data["genres_col"]
    movie_id_col = data["movie_id_col"]

    columns = st.columns(4)

    for index, (_, movie) in enumerate(trending.head(8).iterrows()):
        title = movie.get(title_col, "Unknown Movie")
        genre = movie.get(genres_col, "Cinema") if genres_col else "Cinema"

        average = float(movie.get("average_rating", 0))
        count = int(movie.get("rating_count", 0))

        with columns[index % 4]:
            render_movie_poster(
                title, genre, average, index + 1, count
            )

            if st.button(
                "VIEW MOVIE",
                key=f"home_movie_{index}",
                use_container_width=True
            ):
                st.session_state.selected_movie = {
                    "id": movie.get(movie_id_col) if movie_id_col else None,
                    "title": title
                }

                st.session_state.current_page = "Explore Movies"
                st.rerun()


# ==================================================
# FEATURE STRIP
# ==================================================

def render_feature_strip():
    render_html("""
        <div class="home-feature-strip">
            <div>
                <div class="feature-strip-label">
                    YOUR PERSONAL CINEMA ENGINE
                </div>

                <div class="feature-strip-title">
                    Not just popular. Perfect for your taste.
                </div>

                <div class="feature-strip-desc">
                    Explore intelligent recommendations built
                    from movie content and community preferences.
                </div>
            </div>

            <div style="
                font:700 38px 'Space Grotesk';
                color:#c9f45a;
                opacity:.85;
            ">✳</div>
        </div>
    """)

    if st.button(
        "DISCOVER MY MOVIES  ↗",
        key="home_feature_cta",
        use_container_width=True
    ):
        st.session_state.current_page = "Recommendations"
        st.rerun()


# ==================================================
# MAIN HOME RENDERER
# ==================================================

def render_home(model):
    inject_home_styles()

    if model is None:
        st.error("Recommendation model is not loaded.")
        return

    data = get_home_data(model)

    render_hero(data["total_movies"])
    render_statistics(data)
    render_trending(data)
    render_feature_strip()

    render_html("""
        <div style="
            text-align:center;
            padding:25px 0 5px;
            color:#626c5e;
            font-size:9px;
            letter-spacing:2px;
        ">
            YOUR NEXT OBSESSION IS JUST A DISCOVERY AWAY.
        </div>
    """)
