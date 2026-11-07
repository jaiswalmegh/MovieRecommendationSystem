
"""
CineMatch - Feature Engineering
Converts movie metadata into machine-readable features.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

from config.settings import (
    PROCESSED_DATA_DIR,
    TFIDF_MAX_FEATURES,
    TFIDF_STOP_WORDS,
    TFIDF_NGRAM_RANGE,
)


class MovieFeatureEngineer:
    """Creates content-based features from movie metadata."""

    GENRE_COLUMNS = [
        "unknown",
        "action",
        "adventure",
        "animation",
        "childrens",
        "comedy",
        "crime",
        "documentary",
        "drama",
        "fantasy",
        "film_noir",
        "horror",
        "musical",
        "mystery",
        "romance",
        "sci_fi",
        "thriller",
        "war",
        "western",
    ]

    def __init__(self):
        self.output_dir = Path(PROCESSED_DATA_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            stop_words=TFIDF_STOP_WORDS,
            ngram_range=TFIDF_NGRAM_RANGE,
        )

        self.genre_encoder = MultiLabelBinarizer()

        self.tfidf_matrix = None
        self.genre_matrix = None
        self.combined_features = None

    @staticmethod
    def clean_title(title):
        """Normalize movie title text."""

        if pd.isna(title):
            return ""

        title = str(title).strip()

        # Remove release year suffix such as (1995).
        if title.endswith(")") and "(" in title:
            title = title.rsplit("(", 1)[0].strip()

        return title.lower()

    def extract_genres(self, movies):
        """Extract active genres for every movie."""

        genre_columns = [
            column
            for column in self.GENRE_COLUMNS
            if column in movies.columns
        ]

        if not genre_columns:
            raise ValueError(
                "No genre columns found in the movies dataset."
            )

        genre_lists = []

        for _, movie in movies.iterrows():
            active_genres = [
                genre
                for genre in genre_columns
                if movie[genre] == 1
            ]

            genre_lists.append(active_genres)

        return genre_lists

    def create_text_features(self, movies):
        """Generate TF-IDF features from movie titles."""

        print("Creating TF-IDF text features...")

        titles = movies["title"].apply(self.clean_title)

        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(
            titles
        )

        print(
            f"TF-IDF feature matrix: "
            f"{self.tfidf_matrix.shape}"
        )

        return self.tfidf_matrix

    def create_genre_features(self, movies):
        """Generate binary genre features."""

        print("Creating genre features...")

        genre_lists = self.extract_genres(movies)

        self.genre_matrix = self.genre_encoder.fit_transform(
            genre_lists
        )

        print(
            f"Genre feature matrix: "
            f"{self.genre_matrix.shape}"
        )

        return self.genre_matrix

    def combine_features(self):
        """Combine text and genre feature matrices."""

        if self.tfidf_matrix is None or self.genre_matrix is None:
            raise ValueError(
                "Create text and genre features before combining."
            )

        from scipy.sparse import csr_matrix, hstack

        genre_sparse = csr_matrix(self.genre_matrix)

        self.combined_features = hstack(
            [
                self.tfidf_matrix,
                genre_sparse,
            ],
            format="csr",
        )

        print(
            f"Combined feature matrix: "
            f"{self.combined_features.shape}"
        )

        return self.combined_features

    def save_features(self, movies):
        """Save feature matrix and movie ID mapping."""

        if self.combined_features is None:
            raise ValueError(
                "No combined features available to save."
            )

        from scipy.sparse import save_npz

        feature_path = self.output_dir / "movie_features.npz"
        movie_ids_path = self.output_dir / "movie_feature_ids.npy"
        vectorizer_path = self.output_dir / "tfidf_vectorizer.joblib"
        encoder_path = self.output_dir / "genre_encoder.joblib"

        save_npz(
            feature_path,
            self.combined_features,
        )

        np.save(
            movie_ids_path,
            movies["movie_id"].to_numpy(),
        )

        import joblib

        joblib.dump(
            self.tfidf_vectorizer,
            vectorizer_path,
        )

        joblib.dump(
            self.genre_encoder,
            encoder_path,
        )

        print("\nFeature files saved successfully!")

        return {
            "features": feature_path,
            "movie_ids": movie_ids_path,
            "vectorizer": vectorizer_path,
            "genre_encoder": encoder_path,
        }

    def engineer_features(self, movies):
        """Run the complete feature engineering pipeline."""

        print("\n" + "=" * 50)
        print("       CINEMATCH FEATURE ENGINEERING")
        print("=" * 50)

        movies = movies.copy().reset_index(drop=True)

        self.create_text_features(movies)
        self.create_genre_features(movies)
        self.combine_features()

        saved_files = self.save_features(movies)

        print("\nFeature engineering completed!")
        print(f"Total movies processed: {len(movies)}")
        print(
            f"Total features generated: "
            f"{self.combined_features.shape[1]}"
        )

        return {
            "features": self.combined_features,
            "movie_ids": movies["movie_id"].to_numpy(),
            "saved_files": saved_files,
        }


if __name__ == "__main__":
    movies_path = (
        Path(PROCESSED_DATA_DIR) / "movies_cleaned.csv"
    )

    if not movies_path.exists():
        raise FileNotFoundError(
            "Cleaned movies not found. "
            "Run data_cleaner.py first."
        )

    movies = pd.read_csv(movies_path)

    engineer = MovieFeatureEngineer()

    result = engineer.engineer_features(movies)

    print("\nFinal feature matrix:")
    print(result["features"].shape)
