
"""
CineMatch - Content-Based Recommendation Model

Uses TF-IDF and cosine similarity to recommend movies
based on movie titles and genres.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import (
    PROCESSED_DATA_DIR,
    TOP_N,
)


class ContentBasedRecommender:
    """Content-based movie recommendation engine."""

    def __init__(self):
        self.data_dir = Path(PROCESSED_DATA_DIR)

        self.movies = None
        self.movie_features = None
        self.movie_ids = None

        self.movie_id_to_index = {}
        self.similarity_matrix = None

        self.is_fitted = False

    def load_data(self):
        """Load cleaned movies and engineered features."""

        movies_path = self.data_dir / "movies_cleaned.csv"
        features_path = self.data_dir / "movie_features.npz"
        ids_path = self.data_dir / "movie_feature_ids.npy"

        required_files = [
            movies_path,
            features_path,
            ids_path,
        ]

        for file_path in required_files:
            if not file_path.exists():
                raise FileNotFoundError(
                    f"Required file not found: {file_path}. "
                    "Run data cleaning and feature engineering first."
                )

        self.movies = pd.read_csv(movies_path)

        self.movie_features = load_npz(features_path)

        self.movie_ids = np.load(ids_path)

        if len(self.movies) != len(self.movie_ids):
            raise ValueError(
                "Movie metadata and feature IDs have different lengths."
            )

        if self.movie_features.shape[0] != len(self.movie_ids):
            raise ValueError(
                "Feature matrix and movie IDs have different lengths."
            )

        self.movie_id_to_index = {
            int(movie_id): index
            for index, movie_id in enumerate(self.movie_ids)
        }

        print("Content-based data loaded successfully.")

        return self

    def fit(self, movies=None, features=None, movie_ids=None):
        """
        Initialize the recommendation model.

        Data can either be loaded from processed files or
        provided directly.
        """

        if movies is not None:
            self.movies = movies.copy()

        if features is not None:
            self.movie_features = features

        if movie_ids is not None:
            self.movie_ids = np.asarray(movie_ids)

        if (
            self.movies is None
            or self.movie_features is None
            or self.movie_ids is None
        ):
            self.load_data()

        if len(self.movie_ids) != self.movie_features.shape[0]:
            raise ValueError(
                "Movie IDs do not match the feature matrix."
            )

        if len(self.movie_ids) != len(self.movies):
            raise ValueError(
                "Movie metadata does not match movie IDs."
            )

        self.movie_id_to_index = {
            int(movie_id): index
            for index, movie_id in enumerate(self.movie_ids)
        }

        print("Calculating movie similarity...")

        self.similarity_matrix = cosine_similarity(
            self.movie_features,
            dense_output=True,
        )

        self.is_fitted = True

        print("Content-based model is ready!")

        return self

    def get_movie_index(self, movie_id):
        """Get the internal matrix index for a movie ID."""

        if not self.is_fitted:
            raise RuntimeError(
                "Model is not fitted. Call fit() first."
            )

        movie_id = int(movie_id)

        if movie_id not in self.movie_id_to_index:
            raise ValueError(
                f"Movie ID {movie_id} does not exist."
            )

        return self.movie_id_to_index[movie_id]

    def get_movie_details(self, movie_id):
        """Return metadata for a particular movie."""

        if self.movies is None:
            raise RuntimeError("Movie metadata is not loaded.")

        result = self.movies[
            self.movies["movie_id"] == int(movie_id)
        ]

        if result.empty:
            raise ValueError(
                f"Movie ID {movie_id} not found."
            )

        return result.iloc[0]

    def get_similar_movies(self, movie_id, top_n=None):
        """Find movies most similar to a given movie."""

        if not self.is_fitted:
            raise RuntimeError(
                "Model is not fitted. Call fit() first."
            )

        if top_n is None:
            top_n = TOP_N

        if top_n < 1:
            raise ValueError("top_n must be at least 1.")

        movie_index = self.get_movie_index(movie_id)

        similarity_scores = self.similarity_matrix[movie_index]

        # Exclude the selected movie itself.
        similar_indices = np.argsort(
            similarity_scores
        )[::-1]

        similar_indices = [
            index
            for index in similar_indices
            if index != movie_index
        ][:top_n]

        recommended_ids = self.movie_ids[similar_indices]

        recommendations = self.movies[
            self.movies["movie_id"].isin(recommended_ids)
        ].copy()

        score_map = {
            int(self.movie_ids[index]): float(
                similarity_scores[index]
            )
            for index in similar_indices
        }

        recommendations["similarity_score"] = (
            recommendations["movie_id"]
            .map(score_map)
        )

        recommendations = recommendations.sort_values(
            "similarity_score",
            ascending=False,
        )

        return recommendations.reset_index(drop=True)

    def recommend_by_title(self, title, top_n=None):
        """Recommend movies using a movie title."""

        if self.movies is None:
            raise RuntimeError("Movie metadata is not loaded.")

        matches = self.movies[
            self.movies["title"].str.contains(
                title,
                case=False,
                na=False,
                regex=False,
            )
        ]

        if matches.empty:
            raise ValueError(
                f"No movie found matching '{title}'."
            )

        selected_movie = matches.iloc[0]

        recommendations = self.get_similar_movies(
            selected_movie["movie_id"],
            top_n=top_n,
        )

        return recommendations

    def recommend_by_genre(self, genres, top_n=None):
        """Recommend movies belonging to selected genres."""

        if not self.is_fitted:
            raise RuntimeError(
                "Model is not fitted. Call fit() first."
            )

        if top_n is None:
            top_n = TOP_N

        if isinstance(genres, str):
            genres = [genres]

        selected_genres = [
            genre.lower().strip()
            for genre in genres
        ]

        available_genres = [
            genre
            for genre in selected_genres
            if genre in self.movies.columns
        ]

        if not available_genres:
            raise ValueError(
                "No valid genre columns were provided."
            )

        filtered_movies = self.movies.copy()

        genre_mask = filtered_movies[
            available_genres
        ].eq(1).any(axis=1)

        recommendations = filtered_movies[
            genre_mask
        ].copy()

        recommendations = recommendations.sort_values(
            "title"
        )

        return recommendations.head(top_n).reset_index(drop=True)

    def get_movie_count(self):
        """Return the total number of movies in the model."""

        if self.movie_ids is None:
            return 0

        return len(self.movie_ids)


if __name__ == "__main__":

    print("\n" + "=" * 50)
    print("       CINEMATCH CONTENT-BASED MODEL")
    print("=" * 50)

    recommender = ContentBasedRecommender()

    recommender.fit()

    print(
        f"\nTotal movies in model: "
        f"{recommender.get_movie_count()}"
    )

    # Example movie ID from MovieLens 100K.
    test_movie_id = 1

    print("\nFinding similar movies...")

    try:
        selected_movie = recommender.get_movie_details(
            test_movie_id
        )

        print(f"\nSelected Movie: {selected_movie['title']}")

        recommendations = recommender.get_similar_movies(
            test_movie_id,
            top_n=TOP_N,
        )

        print("\nRecommended Movies:")

        print(
            recommendations[
                ["movie_id", "title", "similarity_score"]
            ].to_string(index=False)
        )

    except ValueError as error:
        print(f"Recommendation error: {error}")
