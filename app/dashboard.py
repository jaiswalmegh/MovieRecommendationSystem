
from pathlib import Path
import sys
import importlib

import streamlit as st
import joblib


# ==================================================
# PROJECT CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_PATH = PROJECT_ROOT / "artifacts" / "hybrid_recommender.joblib"
THEME_PATH = PROJECT_ROOT / "assets" / "styles" / "theme.css"


# ==================================================
# STREAMLIT CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="CineMatch | Discover Your Next Obsession",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


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
# THEME
# ==================================================

def load_theme():
    if THEME_PATH.exists():
        css = THEME_PATH.read_text(encoding="utf-8")
        st.markdown(
            f"<style>\n{css}\n</style>",
            unsafe_allow_html=True
        )
    else:
        st.warning("Theme file not found: assets/styles/theme.css")


load_theme()


# ==================================================
# MODEL
# ==================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None

    try:
        return joblib.load(MODEL_PATH)
    except Exception as error:
        return {"load_error": str(error)}


model = load_model()


# ==================================================
# SESSION STATE
# ==================================================

def initialize_session():
    defaults = {
        "current_page": "Home",
        "selected_movie": None,
        "selected_user": 1,
        "search_query": "",
        "selected_genre": "All"
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_session()


# ==================================================
# NAVIGATION
# ==================================================

NAVIGATION = [
    ("Home", "⌂"),
    ("Explore Movies", "◈"),
    ("Recommendations", "✳"),
    ("Model Insights", "⌁")
]

PAGE_MODULES = {
    "Home": ("app.pages.home", "render_home"),
    "Explore Movies": ("app.pages.explore_movies", "render_explore_movies"),
    "Recommendations": ("app.pages.recommendations", "render_recommendations"),
    "Model Insights": ("app.pages.model_insights", "render_model_insights")
}


# ==================================================
# BRAND
# ==================================================

def render_brand():
    render_html("""
        <div class="cm-brand">
            <div class="cm-brand-icon">CM</div>
            <div>
                <div class="cm-brand-name">CINEMATCH<span>®</span></div>
                <div class="cm-brand-subtitle">YOUR NEXT OBSESSION</div>
            </div>
        </div>
    """)


# ==================================================
# SIDEBAR
# ==================================================

def render_sidebar():
    with st.sidebar:
        render_brand()

        render_html("""
            <div class="cm-sidebar-label">MAIN MENU</div>
        """)

        for page, icon in NAVIGATION:
            active = st.session_state.current_page == page

            if st.button(
                f"{icon}   {page}",
                key=f"nav_{page}",
                use_container_width=True,
                type="primary" if active else "secondary"
            ):
                st.session_state.current_page = page
                st.rerun()

        st.markdown("---")

        online = (
            model is not None
            and not (
                isinstance(model, dict)
                and "load_error" in model
            )
        )

        status = "ONLINE" if online else "OFFLINE"
        color = "#c9f45a" if online else "#ff7777"

        render_html(f"""
            <div class="cm-sidebar-label">PLATFORM STATUS</div>
            <div class="cm-status-panel">
                <div class="cm-status-row">
                    <span class="cm-status-dot"></span>
                    <span>Interface</span>
                    <strong>READY</strong>
                </div>
                <div class="cm-status-row">
                    <span class="cm-status-dot"
                          style="background:{color}"></span>
                    <span>AI Engine</span>
                    <strong style="color:{color}">{status}</strong>
                </div>
            </div>
            <div class="cm-sidebar-footer">
                CINEMATCH / AFTER DARK<br>
                <span>VERSION 1.0.0</span>
            </div>
        """)


# ==================================================
# HEADER
# ==================================================

def render_topbar():
    page = st.session_state.current_page

    render_html(f"""
        <div class="cm-topbar">
            <div>
                <div class="cm-breadcrumb">
                    CINEMATCH / DISCOVER / {page.upper()}
                </div>
                <div class="cm-topbar-heading">{page}</div>
            </div>
            <div class="cm-topbar-right">
                <span class="cm-live-indicator"></span>
                <span>ALL SYSTEMS OPERATIONAL</span>
            </div>
        </div>
    """)


# ==================================================
# MODEL STATUS
# ==================================================

def render_model_error():
    if isinstance(model, dict) and "load_error" in model:
        st.error("Recommendation model could not be loaded.")
        with st.expander("Technical details"):
            st.code(model["load_error"])
        return True

    if model is None:
        render_html("""
            <div class="cm-warning-panel">
                <div class="cm-warning-title">AI ENGINE NOT INITIALIZED</div>
                <p>Start CineMatch using the project launcher.</p>
                <code>python main.py</code>
            </div>
        """)
        return True

    return False


# ==================================================
# PAGE ROUTER
# ==================================================

def render_page():
    page = st.session_state.current_page
    module_info = PAGE_MODULES.get(page)

    if not module_info:
        st.error("Page not found.")
        return

    module_name, function_name = module_info

    try:
        page_module = importlib.import_module(module_name)
        render_function = getattr(page_module, function_name)
        render_function(model)

    except Exception as error:
        st.error(f"Unable to load {page}.")

        with st.expander("Technical details"):
            st.exception(error)


# ==================================================
# FOOTER
# ==================================================

def render_footer():
    render_html("""
        <div class="cm-app-footer">
            <span>© 2026 CINEMATCH</span>
            <span>BUILT FOR MOVIE LOVERS. POWERED BY AI.</span>
        </div>
    """)


# ==================================================
# MAIN
# ==================================================

def main():
    render_sidebar()

    render_topbar()

    if not render_model_error():
        render_page()

    render_footer()


if __name__ == "__main__":
    main()
