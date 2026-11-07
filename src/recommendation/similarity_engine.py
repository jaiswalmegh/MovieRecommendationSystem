
"""
CineMatch - Similarity Engine

Handles movie-to-movie similarity calculations
using cosine similarity and engineered features.
"""

import numpy as np
import pandas as pd

from pathlib import Path
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import PROCESSED_DATA_DIR, TOP_N


class SimilarityEngine:
    """Provides movie similarity and comparison operations."""

    def __init__(self):
        self.data_dir = Path(PROCESSED_DATA_DIR)

        self.movies = None
        self.features = None
        self.movie_ids = None

        self.movie_id_to_index = {}
        self.similarity_matrix = None

        self.is_loaded = False

    def load_data(self):
        """Load processed movie metadata and feature matrix."""

        movies_path = self.data_dir / "movies_cleaned.csv"
        features_path = self.data_dir / "movie_features.npz"
        ids_path = self.data_dir / "movie_feature_ids.npy"

        required_files = [
            movies_path,
            features_path,
            ids_path,
        ]

        for path in required_files:
            if not path.exists():
                raise FileNotFoundError(
                    f"Required file not found: {path}. "
                    "Run data cleaning and feature engineering first."
                )

        self.movies = pd.read_csv(movies_path)
        self.features = load_npz(features_path)
        self.movie_ids = np.load(ids_path)

        if len(self.movie_ids) != self.features.shape[0]:
            raise ValueError(
                "Movie IDs and feature matrix dimensions do not match."
            )

        if len(self.movies) != len(self.movie_ids):
            raise ValueError(
                "Movie metadata and feature IDs do not match."
            )

        self.movie_id_to_index = {
            int(movie_id): index
            for index, movie_id in enumerate(self.movie_ids)
        }

        self.is_loaded = True

        print("Similarity engine data loaded successfully.")

        return self

    def build_similarity_matrix(self):
        """Calculate cosine similarity between all movies."""

        if not self.is_loaded:
            self.load_data()

        print("Building movie similarity matrix...")

        self.similarity_matrix = cosine_similarity(
            self.features,
            dense_output=True,
        )

        print(
            "Similarity matrix shape:",
            self.similarity_matrix.shape,
        )

        return self.similarity_matrix

    def _ensure_similarity_ready(self):
        """Ensure similarity matrix is available."""

        if self.similarity_matrix is None:
            self.build_similarity_matrix()

    def get_movie_index(self, movie_id):
        """Get the feature matrix index for a movie."""

        if not self.is_loaded:
            self.load_data()

        movie_id = int(movie_id)

        if movie_id not in self.movie_id_to_index:
            raise ValueError(
                f"Movie ID {movie_id} does not exist."
            )

        return self.movie_id_to_index[movie_id]

    def get_similarity(self, movie_id_1, movie_id_2):
        """Calculate similarity between two movies."""

        self._ensure_similarity_ready()

        index_1 = self.get_movie_index(movie_id_1)
        index_2 = self.get_movie_index(movie_id_2)

        return float(
            self.similarity_matrix[index_1, index_2]
        )

    def get_similar_movies(self, movie_id, top_n=None):
        """Return the most similar movies."""

        self._ensure_similarity_ready()

        if top_n is None:
            top_n = TOP_N

        if top_n < 1:
            raise ValueError("top_n must be at least 1.")

        movie_index = self.get_movie_index(movie_id)

        scores = self.similarity_matrix[movie_index]

        sorted_indices = np.argsort(scores)[::-1]

        sorted_indices = [
            index
            for index in sorted_indices
            if index != movie_index
        ][:top_n]

        selected_ids = self.movie_ids[sorted_indices]

        recommendations = self.movies[
            self.movies["movie_id"].isin(selected_ids)
        ].copy()

        score_map = {
            int(self.movie_ids[index]): float(scores[index])
            for index in sorted_indices
        }

        recommendations["similarity_score"] = (
            recommendations["movie_id"].map(score_map)
        )

        return (
            recommendations
            .sort_values("similarity_score", ascending=False)
            .reset_index(drop=True)
        )

    def get_similarity_dataframe(self, movie_ids):
        """Generate a similarity table for selected movies."""

        self._ensure_similarity_ready()

        movie_ids = [int(movie_id) for movie_id in movie_ids]

        indices = [
            self.get_movie_index(movie_id)
            for movie_id in movie_ids
        ]

        selected_similarity = self.similarity_matrix[
            np.ix_(indices, indices)
        ]

        return pd.DataFrame(
            selected_similarity,
            index=movie_ids,
            columns=movie_ids,
        )

    def find_most_similar_pair(self):
        """Find the most similar distinct movie pair."""

        self._ensure_similarity_ready()

        if len(self.movie_ids) < 2:
            raise ValueError(
                "At least two movies are required."
            )

        matrix = self.similarity_matrix.copy()

        np.fill_diagonal(matrix, -1)

        flat_index = np.argmax(matrix)

        row, column = np.unravel_index(
            flat_index,
            matrix.shape,
        )

        return {
            "movie_1_id": int(self.movie_ids[row]),
            "movie_2_id": int(self.movie_ids[column]),
            "similarity": float(matrix[row, column]),
        }

    def get_movie_details(self, movie_id):
        """Return movie metadata."""

        if not self.is_loaded:
            self.load_data()

        result = self.movies[
            self.movies["movie_id"] == int(movie_id)
        ]

        if result.empty:
            raise ValueError(
                f"Movie ID {movie_id} not found."
            )

        return result.iloc[0]


if __name__ == "__main__":

    print("\n" + "=" * 55)
    print("       CINEMATCH SIMILARITY ENGINE")
    print("=" * 55)

    engine = SimilarityEngine()

    engine.load_data()
    engine.build_similarity_matrix()

    test_movie_id = 1

    try:
        movie = engine.get_movie_details(test_movie_id)

        print(f"\nSelected Movie: {movie['title']}")

        recommendations = engine.get_similar_movies(
            test_movie_id,
            top_n=TOP_N,
        )

        print("\nMost Similar Movies:")

        print(
            recommendations[
                ["movie_id", "title", "similarity_score"]
            ].to_string(index=False)
        )

        pair = engine.find_most_similar_pair()

        print("\nMost Similar Movie Pair:")
        print(pair)

    except ValueError as error:
        print(f"Error: {error}")
