
from pathlib import Path


# ==========================================
# PROJECT CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROJECT_NAME = "CineMatch"
PROJECT_VERSION = "1.0.0"
PROJECT_DESCRIPTION = "Hybrid Movie Recommendation System"


# ==========================================
# DIRECTORY PATHS
# ==========================================

CONFIG_DIR = PROJECT_ROOT / "config"

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

SRC_DIR = PROJECT_ROOT / "src"

APP_DIR = PROJECT_ROOT / "app"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

TESTS_DIR = PROJECT_ROOT / "tests"

ASSETS_DIR = PROJECT_ROOT / "assets"
STYLES_DIR = ASSETS_DIR / "styles"
IMAGES_DIR = ASSETS_DIR / "images"


# ==========================================
# MOVIELENS DATASET CONFIGURATION
# ==========================================

DATASET_NAME = "MovieLens 100K"

DATASET_URL = (
    "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
)

DATASET_ARCHIVE = RAW_DATA_DIR / "ml-100k.zip"

DATASET_EXTRACTED_DIR = RAW_DATA_DIR / "ml-100k"

MOVIES_FILE = DATASET_EXTRACTED_DIR / "u.item"

RATINGS_FILE = DATASET_EXTRACTED_DIR / "u.data"

USERS_FILE = DATASET_EXTRACTED_DIR / "u.user"

GENRES_FILE = DATASET_EXTRACTED_DIR / "u.genre"


# ==========================================
# PROCESSED DATA FILES
# ==========================================

PROCESSED_MOVIES_FILE = (
    PROCESSED_DATA_DIR / "movies_processed.csv"
)

PROCESSED_RATINGS_FILE = (
    PROCESSED_DATA_DIR / "ratings_processed.csv"
)

USER_ITEM_MATRIX_FILE = (
    PROCESSED_DATA_DIR / "user_item_matrix.csv"
)


# ==========================================
# MACHINE LEARNING CONFIGURATION
# ==========================================

RANDOM_STATE = 42

TEST_SIZE = 0.20

MINIMUM_RATING = 1

MAXIMUM_RATING = 5

TOP_N_RECOMMENDATIONS = 10


# ==========================================
# BACKWARD-COMPATIBLE CONSTANT ALIASES
# ==========================================

# Rating configuration
RATING_MIN = MINIMUM_RATING
RATING_MAX = MAXIMUM_RATING

# Recommendation count
TOP_N = TOP_N_RECOMMENDATIONS


# ==========================================
# CONTENT-BASED FILTERING
# ==========================================

TFIDF_MAX_FEATURES = 5000

TFIDF_STOP_WORDS = "english"

TFIDF_NGRAM_RANGE = (1, 2)


# ==========================================
# COLLABORATIVE FILTERING
# ==========================================

COLLABORATIVE_NEIGHBORS = 10

MINIMUM_USER_RATINGS = 5

MINIMUM_MOVIE_RATINGS = 5


# Backward-compatible aliases
COLLABORATIVE_N_NEIGHBORS = COLLABORATIVE_NEIGHBORS

MIN_USER_RATINGS = MINIMUM_USER_RATINGS

MIN_MOVIE_RATINGS = MINIMUM_MOVIE_RATINGS


# ==========================================
# HYBRID RECOMMENDATION CONFIGURATION
# ==========================================

CONTENT_WEIGHT = 0.5

COLLABORATIVE_WEIGHT = 0.5


# Backward-compatible aliases
HYBRID_CONTENT_WEIGHT = CONTENT_WEIGHT

HYBRID_COLLABORATIVE_WEIGHT = COLLABORATIVE_WEIGHT


# ==========================================
# EVALUATION CONFIGURATION
# ==========================================

EVALUATION_K_VALUES = [5, 10, 15, 20]

RECOMMENDATION_THRESHOLD = 4.0


# ==========================================
# APPLICATION CONFIGURATION
# ==========================================

APP_TITLE = "CineMatch | Movie Recommendation System"

APP_LAYOUT = "wide"

APP_ICON = "🎬"


# ==========================================
# DIRECTORY INITIALIZATION
# ==========================================

def create_project_directories():
    """Create all required project directories."""

    directories = [
        CONFIG_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        NOTEBOOKS_DIR,
        SRC_DIR,
        APP_DIR,
        ARTIFACTS_DIR,
        TESTS_DIR,
        IMAGES_DIR,
        STYLES_DIR,
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True
        )


# ==========================================
# CONFIGURATION VALIDATION
# ==========================================

def validate_configuration():
    """Validate important project configuration values."""

    assert 0 < TEST_SIZE < 1

    assert RATING_MIN < RATING_MAX

    assert TOP_N >= 1

    assert TFIDF_MAX_FEATURES > 0

    assert HYBRID_CONTENT_WEIGHT >= 0

    assert HYBRID_COLLABORATIVE_WEIGHT >= 0

    assert (
        HYBRID_CONTENT_WEIGHT
        + HYBRID_COLLABORATIVE_WEIGHT
    ) > 0

    assert COLLABORATIVE_NEIGHBORS >= 1

    assert RECOMMENDATION_THRESHOLD >= RATING_MIN

    assert RECOMMENDATION_THRESHOLD <= RATING_MAX

    return True


# ==========================================
# INITIALIZATION
# ==========================================

if __name__ == "__main__":

    create_project_directories()

    validate_configuration()

    print("=" * 45)
    print(f"{PROJECT_NAME} Configuration Initialized")
    print("=" * 45)

    print(f"Version: {PROJECT_VERSION}")
    print(f"Project location: {PROJECT_ROOT}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Top recommendations: {TOP_N}")
    print("Configuration validation: PASSED")
