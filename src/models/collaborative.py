
"""
CineMatch - Collaborative Filtering Model

Uses User-User Collaborative Filtering to recommend
movies based on similar users' rating patterns.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

from config.settings import (
    PROCESSED_DATA_DIR,
    TOP_N,
)


class CollaborativeRecommender:
    """User-based collaborative filtering recommendation engine."""

    def __init__(
        self,
        n_neighbors=10,
        min_user_ratings=5,
        min_movie_ratings=5,
    ):
        self.data_dir = Path(PROCESSED_DATA_DIR)

        self.n_neighbors = n_neighbors
        self.min_user_ratings = min_user_ratings
        self.min_movie_ratings = min_movie_ratings

        self.ratings = None
        self.movies = None

        self.user_item_matrix = None
        self.user_similarity_matrix = None

        self.user_ids = None
        self.movie_ids = None

        self.user_id_to_index = {}
        self.movie_id_to_index = {}

        self.is_fitted = False

    def load_data(self):
        """Load cleaned ratings and movie metadata."""

        ratings_path = self.data_dir / "ratings_cleaned.csv"
        movies_path = self.data_dir / "movies_cleaned.csv"

        if not ratings_path.exists():
            raise FileNotFoundError(
                f"Ratings file not found: {ratings_path}. "
                "Run data cleaning first."
            )

        if not movies_path.exists():
            raise FileNotFoundError(
                f"Movies file not found: {movies_path}. "
                "Run data cleaning first."
            )

        self.ratings = pd.read_csv(ratings_path)
        self.movies = pd.read_csv(movies_path)

        print("Collaborative filtering data loaded.")

        return self

    def prepare_user_item_matrix(self):
        """Create the user-movie rating matrix."""

        if self.ratings is None:
            raise RuntimeError(
                "Ratings data not loaded."
            )

        print("Creating user-item rating matrix...")

        ratings = self.ratings.copy()

        user_counts = ratings.groupby("user_id").size()
        movie_counts = ratings.groupby("movie_id").size()

        valid_users = user_counts[
            user_counts >= self.min_user_ratings
        ].index

        valid_movies = movie_counts[
            movie_counts >= self.min_movie_ratings
        ].index

        ratings = ratings[
            ratings["user_id"].isin(valid_users)
            & ratings["movie_id"].isin(valid_movies)
        ]

        if ratings.empty:
            raise ValueError(
                "No ratings remain after applying minimum "
                "user and movie rating thresholds."
            )

        self.user_item_matrix = ratings.pivot_table(
            index="user_id",
            columns="movie_id",
            values="rating",
            aggfunc="mean",
            fill_value=0,
        )

        self.user_ids = self.user_item_matrix.index.to_numpy()
        self.movie_ids = self.user_item_matrix.columns.to_numpy()

        self.user_id_to_index = {
            int(user_id): index
            for index, user_id in enumerate(self.user_ids)
        }

        self.movie_id_to_index = {
            int(movie_id): index
            for index, movie_id in enumerate(self.movie_ids)
        }

        print(
            f"User-item matrix shape: "
            f"{self.user_item_matrix.shape}"
        )

        return self.user_item_matrix

    def calculate_user_similarity(self):
        """Calculate cosine similarity between users."""

        if self.user_item_matrix is None:
            raise RuntimeError(
                "Create the user-item matrix first."
            )

        print("Calculating user similarity...")

        self.user_similarity_matrix = cosine_similarity(
            self.user_item_matrix.to_numpy(),
            dense_output=True,
        )

        np.fill_diagonal(
            self.user_similarity_matrix,
            0,
        )

        print("User similarity calculation completed.")

        return self.user_similarity_matrix

    def fit(self, ratings=None, movies=None):
        """Train the collaborative filtering model."""

        if ratings is not None:
            self.ratings = ratings.copy()

        if movies is not None:
            self.movies = movies.copy()

        if self.ratings is None or self.movies is None:
            self.load_data()

        self.prepare_user_item_matrix()
        self.calculate_user_similarity()

        self.is_fitted = True

        print("\nCollaborative model is ready!")

        return self

    def get_similar_users(self, user_id, n_neighbors=None):
        """Find the most similar users."""

        if not self.is_fitted:
            raise RuntimeError(
                "Model is not fitted. Call fit() first."
            )

        user_id = int(user_id)

        if user_id not in self.user_id_to_index:
            raise ValueError(
                f"User ID {user_id} is not available in the model."
            )

        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        if n_neighbors < 1:
            raise ValueError(
                "n_neighbors must be at least 1."
            )

        user_index = self.user_id_to_index[user_id]

        similarities = self.user_similarity_matrix[user_index]

        similar_indices = np.argsort(similarities)[::-1]

        similar_indices = [
            index
            for index in similar_indices
            if similarities[index] > 0
        ][:n_neighbors]

        similar_users = pd.DataFrame({
            "user_id": self.user_ids[similar_indices],
            "similarity": similarities[similar_indices],
        })

        return similar_users.reset_index(drop=True)

    def predict_rating(self, user_id, movie_id):
        """
        Predict a user's rating for a movie using
        similarity-weighted ratings from neighboring users.
        """

        if not self.is_fitted:
            raise RuntimeError(
                "Model is not fitted."
            )

        user_id = int(user_id)
        movie_id = int(movie_id)

        if user_id not in self.user_id_to_index:
            raise ValueError(
                f"Unknown user ID: {user_id}"
            )

        if movie_id not in self.movie_id_to_index:
            return np.nan

        user_index = self.user_id_to_index[user_id]
        movie_index = self.movie_id_to_index[movie_id]

        similarities = self.user_similarity_matrix[user_index]

        movie_ratings = self.user_item_matrix.iloc[
            :, movie_index
        ].to_numpy()

        rated_indices = np.where(movie_ratings > 0)[0]

        rated_indices = rated_indices[
            rated_indices != user_index
        ]

        if len(rated_indices) == 0:
            return np.nan

        sorted_indices = rated_indices[
            np.argsort(similarities[rated_indices])[::-1]
        ]

        sorted_indices = sorted_indices[
            similarities[sorted_indices] > 0
        ][:self.n_neighbors]

        if len(sorted_indices) == 0:
            return np.nan

        neighbor_similarities = similarities[sorted_indices]
        neighbor_ratings = movie_ratings[sorted_indices]

        total_similarity = np.sum(
            np.abs(neighbor_similarities)
        )

        if total_similarity == 0:
            return np.nan

        predicted_rating = np.dot(
            neighbor_similarities,
            neighbor_ratings,
        ) / total_similarity

        return float(
            np.clip(predicted_rating, 1, 5)
        )

    def get_recommendations(self, user_id, top_n=None):
        """Generate personalized recommendations for a user."""

        if not self.is_fitted:
            raise RuntimeError(
                "Model is not fitted. Call fit() first."
            )

        if top_n is None:
            top_n = TOP_N

        if top_n < 1:
            raise ValueError("top_n must be at least 1.")

        user_id = int(user_id)

        if user_id not in self.user_id_to_index:
            raise ValueError(
                f"User ID {user_id} is not available in the model."
            )

        user_index = self.user_id_to_index[user_id]

        user_ratings = self.user_item_matrix.iloc[
            user_index
        ]

        watched_movies = set(
            user_ratings[user_ratings > 0].index
        )

        similar_users = self.get_similar_users(
            user_id,
            self.n_neighbors,
        )

        if similar_users.empty:
            return pd.DataFrame(
                columns=[
                    "movie_id",
                    "title",
                    "predicted_rating",
                ]
            )

        neighbor_ids = similar_users["user_id"].to_numpy()

        neighbor_weights = similar_users[
            "similarity"
        ].to_numpy()

        neighbor_matrix = self.user_item_matrix.loc[
            neighbor_ids
        ].to_numpy()

        weighted_ratings = (
            neighbor_matrix * neighbor_weights[:, np.newaxis]
        )

        rated_mask = neighbor_matrix > 0

        numerator = weighted_ratings.sum(axis=0)

        denominator = (
            rated_mask * neighbor_weights[:, np.newaxis]
        ).sum(axis=0)

        with np.errstate(
            divide="ignore",
            invalid="ignore",
        ):
            predictions = np.divide(
                numerator,
                denominator,
                out=np.zeros_like(numerator, dtype=float),
                where=denominator > 0,
            )

        candidates = []

        for index, movie_id in enumerate(self.movie_ids):

            if movie_id in watched_movies:
                continue

            if denominator[index] <= 0:
                continue

            candidates.append({
                "movie_id": int(movie_id),
                "predicted_rating": float(
                    np.clip(predictions[index], 1, 5)
                ),
            })

        if not candidates:
            return pd.DataFrame(
                columns=[
                    "movie_id",
                    "title",
                    "predicted_rating",
                ]
            )

        recommendations = pd.DataFrame(candidates)

        recommendations = recommendations.sort_values(
            "predicted_rating",
            ascending=False,
        ).head(top_n)

        recommendations = recommendations.merge(
            self.movies,
            on="movie_id",
            how="left",
        )

        columns = [
            "movie_id",
            "title",
            "predicted_rating",
        ]

        return recommendations[
            columns
        ].reset_index(drop=True)

    def get_model_statistics(self):
        """Return basic collaborative model statistics."""

        if not self.is_fitted:
            raise RuntimeError(
                "Model is not fitted."
            )

        return {
            "total_users": len(self.user_ids),
            "total_movies": len(self.movie_ids),
            "total_ratings": int(
                (self.user_item_matrix > 0).sum().sum()
            ),
            "matrix_shape": self.user_item_matrix.shape,
            "neighbors": self.n_neighbors,
        }


if __name__ == "__main__":

    print("\n" + "=" * 55)
    print("       CINEMATCH COLLABORATIVE FILTERING")
    print("=" * 55)

    recommender = CollaborativeRecommender(
        n_neighbors=10,
        min_user_ratings=5,
        min_movie_ratings=5,
    )

    recommender.fit()

    statistics = recommender.get_model_statistics()

    print("\nModel Statistics:")

    for key, value in statistics.items():
        print(f"{key}: {value}")

    test_user_id = 1

    print(f"\nFinding recommendations for User {test_user_id}...")

    try:
        similar_users = recommender.get_similar_users(
            test_user_id
        )

        print("\nSimilar Users:")
        print(similar_users.to_string(index=False))

        recommendations = recommender.get_recommendations(
            test_user_id,
            top_n=TOP_N,
        )

        print("\nPersonalized Recommendations:")
        print(recommendations.to_string(index=False))

    except ValueError as error:
        print(f"Recommendation error: {error}")
