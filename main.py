
"""
CineMatch - Unified Application Launcher

Run the complete CineMatch application using:

    python main.py

Automatically:
1. Initializes project configuration.
2. Checks the trained model.
3. Trains the model if unavailable.
4. Launches the Streamlit dashboard.
"""

import sys
import time
import subprocess
import webbrowser

from pathlib import Path

from config.settings import (
    PROJECT_NAME,
    PROJECT_VERSION,
    ARTIFACTS_DIR,
    create_project_directories,
    validate_configuration,
)

from src.utils.logger import get_logger


# ==========================================
# PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    Path(ARTIFACTS_DIR)
    / "hybrid_recommender.joblib"
)

DASHBOARD_PATH = (
    PROJECT_ROOT
    / "app"
    / "dashboard.py"
)

logger = get_logger("CineMatch.Launcher")


# ==========================================
# INITIALIZATION
# ==========================================

def initialize_project():

    print("\n" + "=" * 60)
    print(f"       {PROJECT_NAME} v{PROJECT_VERSION}")
    print("       HYBRID MOVIE RECOMMENDATION SYSTEM")
    print("=" * 60)

    create_project_directories()
    validate_configuration()

    print("\n[OK] Project configuration initialized.")

    logger.info("Project initialized.")


# ==========================================
# MODEL CHECK
# ==========================================

def is_model_available():

    if MODEL_PATH.exists():

        print("\n[OK] Trained recommendation model found.")

        logger.info(
            "Existing model found: %s",
            MODEL_PATH,
        )

        return True

    print("\n[INFO] Trained model not found.")

    return False


# ==========================================
# MODEL TRAINING
# ==========================================

def train_model():

    print("\n" + "=" * 60)
    print("       TRAINING CINEMATCH MODEL")
    print("=" * 60)

    print("\nThis may take a few minutes...")

    try:

        from src.data.data_loader import MovieLensDataLoader
        from src.data.data_cleaner import MovieLensDataCleaner
        from src.data.feature_engineering import MovieFeatureEngineer
        from src.models.hybrid_model import HybridRecommender

        # Load dataset
        print("\n[1/4] Loading dataset...")

        loader = MovieLensDataLoader()

        raw_data = loader.load_all_data()

        # Clean dataset
        print("\n[2/4] Cleaning dataset...")

        cleaner = MovieLensDataCleaner()

        cleaned_data = cleaner.clean_all_data(
            raw_data
        )

        # Generate features
        print("\n[3/4] Generating features...")

        engineer = MovieFeatureEngineer()

        feature_data = engineer.engineer_features(
            cleaned_data["movies"]
        )

        # Train model
        print("\n[4/4] Training hybrid recommendation model...")

        model = HybridRecommender(
            content_weight=0.5,
            collaborative_weight=0.5,
            n_neighbors=10,
        )

        model.fit(
            movies=cleaned_data["movies"],
            ratings=cleaned_data["ratings"],
            features=feature_data["features"],
            movie_ids=feature_data["movie_ids"],
        )

        import joblib

        joblib.dump(
            model,
            MODEL_PATH,
        )

        print("\n[OK] Model trained and saved successfully.")

        logger.info(
            "Model trained and saved: %s",
            MODEL_PATH,
        )

        return True

    except Exception:

        logger.exception(
            "Model training failed."
        )

        print(
            "\n[ERROR] Model training failed. "
            "Check the error above."
        )

        return False


# ==========================================
# DASHBOARD VALIDATION
# ==========================================

def validate_dashboard():

    if not DASHBOARD_PATH.exists():

        print(
            f"\n[ERROR] Dashboard file not found: "
            f"{DASHBOARD_PATH}"
        )

        return False

    return True


# ==========================================
# STREAMLIT LAUNCHER
# ==========================================

def launch_dashboard():

    print("\n" + "=" * 60)
    print("       LAUNCHING CINEMATCH DASHBOARD")
    print("=" * 60)

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(DASHBOARD_PATH),
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
    ]

    try:

        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
        )

        print("\n[OK] Streamlit server is starting.")

        print(
            "\nCineMatch will open in your browser shortly."
        )

        print(
            "\nPress CTRL+C in this terminal to stop CineMatch."
        )

        time.sleep(3)

        webbrowser.open(
            "http://localhost:8501"
        )

        logger.info(
            "Streamlit dashboard launched."
        )

        process.wait()

        return 0

    except KeyboardInterrupt:

        print("\n\nCineMatch shutdown requested.")

        logger.info(
            "CineMatch dashboard stopped by user."
        )

        return 0

    except Exception:

        logger.exception(
            "Dashboard launch failed."
        )

        print(
            "\n[ERROR] Unable to launch dashboard."
        )

        return 1


# ==========================================
# MAIN APPLICATION
# ==========================================

def main():

    initialize_project()

    if not validate_dashboard():

        return 1

    if not is_model_available():

        success = train_model()

        if not success:
            return 1

    return launch_dashboard()


# ==========================================
# ENTRY POINT
# ==========================================

if __name__ == "__main__":

    sys.exit(main())
