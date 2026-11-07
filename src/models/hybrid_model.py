
"""
CineMatch - Hybrid Recommendation Model

Combines Content-Based and Collaborative Filtering
to generate personalized movie recommendations.
"""

import numpy as np
import pandas as pd

from src.models.content_based import ContentBasedRecommender
from src.models.collaborative import CollaborativeRecommender

from config.settings import (
    TOP_N,
    HYBRID_CONTENT_WEIGHT,
    HYBRID_COLLABORATIVE_WEIGHT,
    RECOMMENDATION_THRESHOLD,
)


class HybridRecommender:
    """
    Hybrid recommendation engine combining:
    1. Content-Based Filtering
    2. Collaborative Filtering
    """

    def __init__(
        self,
        content_weight=HYBRID_CONTENT_WEIGHT,
        collaborative_weight=HYBRID_COLLABORATIVE_WEIGHT,
        n_neighbors=10,
    ):
        self.content_weight = content_weight
        self.collaborative_weight = collaborative_weight

        self.n_neighbors = n_neighbors

        self.content_model = ContentBasedRecommender()

        self.collaborative_model = CollaborativeRecommender(
            n_neighbors=n_neighbors
        )

        self.movies = None
        self.ratings = None

        self.is_fitted = False

        self._validate_weights()

    def _validate_weights(self):
        """Validate hybrid model weights."""

        if self.content_weight < 0:
            raise ValueError(
                "Content weight cannot be negative."
            )

        if self.collaborative_weight < 0:
            raise ValueError(
                "Collaborative weight cannot be negative."
            )

        total_weight = (
            self.content_weight
            + self.collaborative_weight
        )

        if total_weight <= 0:
            raise ValueError(
                "At least one recommendation weight must be positive."
            )

        # Normalize weights to sum to 1.
        self.content_weight /= total_weight
        self.collaborative_weight /= total_weight

    def fit(self, movies=None, ratings=None, features=None, movie_ids=None):
        """Train both recommendation models."""

        print("\n" + "=" * 55)
        print("       TRAINING CINEMATCH HYBRID MODEL")
        print("=" * 55)

        if movies is None:
            self.content_model.load_data()
            self.movies = self.content_model.movies.copy()
        else:
            self.movies = movies.copy()

        if ratings is None:
            self.collaborative_model.load_data()
            self.ratings = self.collaborative_model.ratings.copy()
        else:
            self.ratings = ratings.copy()

        self.content_model.fit(
            movies=self.movies,
            features=features,
            movie_ids=movie_ids,
        )

        self.collaborative_model.fit(
            ratings=self.ratings,
            movies=self.movies,
        )

        self.is_fitted = True

        print("\nHybrid model trained successfully!")

        print(
            f"Content weight: "
            f"{self.content_weight:.2f}"
        )

        print(
            f"Collaborative weight: "
            f"{self.collaborative_weight:.2f}"
        )

        return self

    def _normalize_scores(self, scores):
        """Normalize scores between 0 and 1."""

        scores = np.asarray(scores, dtype=float)

        if len(scores) == 0:
            return scores

        minimum = np.min(scores)
        maximum = np.max(scores)

        if maximum == minimum:
            return np.ones_like(scores)

        return (scores - minimum) / (maximum - minimum)

    def get_content_scores(self, movie_id, candidate_ids):
        """Calculate content similarity scores."""

        movie_index = self.content_model.get_movie_index(
            movie_id
        )

        candidate_indices = [
            self.content_model.get_movie_index(candidate_id)
            for candidate_id in candidate_ids
        ]

        scores = self.content_model.similarity_matrix[
            movie_index,
            candidate_indices,
        ]

        return np.asarray(scores, dtype=float)

    def get_collaborative_scores(self, user_id, candidate_ids):
        """Calculate collaborative predicted ratings."""

        scores = []

        for movie_id in candidate_ids:
            predicted_rating = (
                self.collaborative_model.predict_rating(
                    user_id,
                    movie_id,
                )
            )

            scores.append(predicted_rating)

        return np.asarray(scores, dtype=float)

    def recommend_for_user(
        self,
        user_id,
        top_n=None,
        include_scores=True,
    ):
        """
        Generate personalized hybrid recommendations.

        Combines normalized content and collaborative scores.
        Content score is based on the user's rating history.
        """

        if not self.is_fitted:
            raise RuntimeError(
                "Model is not fitted. Call fit() first."
            )

        if top_n is None:
            top_n = TOP_N

        if top_n < 1:
            raise ValueError("top_n must be at least 1.")

        user_id = int(user_id)

        if user_id not in self.collaborative_model.user_id_to_index:
            raise ValueError(
                f"User ID {user_id} is not available."
            )

        ratings = self.ratings[
            self.ratings["user_id"] == user_id
        ]

        if ratings.empty:
            raise ValueError(
                f"No rating history found for user {user_id}."
            )

        # Movies already rated by the user.
        watched_movies = set(
            ratings["movie_id"].astype(int)
        )

        # Use positively rated movies as content preferences.
        liked_movies = ratings[
            ratings["rating"] >= RECOMMENDATION_THRESHOLD
        ]

        # Fallback: use all rated movies if no positive ratings exist.
        if liked_movies.empty:
            liked_movies = ratings

        liked_ids = liked_movies[
            "movie_id"
        ].astype(int).tolist()

        liked_ratings = liked_movies[
            "rating"
        ].to_numpy(dtype=float)

        candidate_ids = [
            int(movie_id)
            for movie_id in self.movies["movie_id"]
            if int(movie_id) not in watched_movies
        ]

        if not candidate_ids:
            return pd.DataFrame()

        # Build user preference scores from liked movies.
        content_score_map = {}

        liked_weights = np.maximum(
            liked_ratings - RECOMMENDATION_THRESHOLD + 1,
            0.1,
        )

        for start in range(0, len(candidate_ids), 500):
            batch_ids = candidate_ids[start:start + 500]

            batch_indices = [
                self.content_model.get_movie_index(movie_id)
                for movie_id in batch_ids
                if movie_id in self.content_model.movie_id_to_index
            ]

            valid_batch_ids = [
                movie_id
                for movie_id in batch_ids
                if movie_id in self.content_model.movie_id_to_index
            ]

            if not valid_batch_ids:
                continue

            similarity_scores = np.zeros(len(valid_batch_ids))

            for liked_id, weight in zip(
                liked_ids,
                liked_weights,
            ):
                liked_index = (
                    self.content_model.get_movie_index(liked_id)
                )

                candidate_indices = [
                    self.content_model.get_movie_index(movie_id)
                    for movie_id in valid_batch_ids
                ]

                similarities = (
                    self.content_model.similarity_matrix[
                        liked_index,
                        candidate_indices,
                    ]
                )

                similarity_scores += similarities * weight

            similarity_scores /= liked_weights.sum()

            for movie_id, score in zip(
                valid_batch_ids,
                similarity_scores,
            ):
                content_score_map[movie_id] = float(score)

        # Collaborative predicted rating scores.
        collaborative_score_map = {}

        for movie_id in candidate_ids:
            predicted_rating = (
                self.collaborative_model.predict_rating(
                    user_id,
                    movie_id,
                )
            )

            collaborative_score_map[movie_id] = (
                float(predicted_rating)
                if not np.isnan(predicted_rating)
                else np.nan
            )

        results = []

        for movie_id in candidate_ids:

            content_score = content_score_map.get(
                movie_id,
                0.0,
            )

            collaborative_score = (
                collaborative_score_map.get(
                    movie_id,
                    np.nan,
                )
            )

            # Normalize collaborative ratings from 1-5 to 0-1.
            if not np.isnan(collaborative_score):
                collaborative_score = (
                    collaborative_score - 1
                ) / 4

            # If collaborative score is unavailable,
            # use content-based recommendation only.
            if np.isnan(collaborative_score):
                final_score = content_score

            else:
                final_score = (
                    self.content_weight * content_score
                    + self.collaborative_weight * collaborative_score
                )

            results.append({
                "movie_id": movie_id,
                "content_score": content_score,
                "collaborative_score": collaborative_score,
                "hybrid_score": float(final_score),
            })

        recommendations = pd.DataFrame(results)

        recommendations = recommendations.merge(
            self.movies,
            on="movie_id",
            how="left",
        )

        recommendations = recommendations.sort_values(
            "hybrid_score",
            ascending=False,
        ).head(top_n)

        output_columns = [
            "movie_id",
            "title",
            "hybrid_score",
        ]

        if include_scores:
            output_columns.extend([
                "content_score",
                "collaborative_score",
            ])

        return recommendations[
            output_columns
        ].reset_index(drop=True)

    def recommend_similar_movies(
        self,
        movie_id,
        top_n=None,
    ):
        """Generate content-based similar movie recommendations."""

        if not self.is_fitted:
            raise RuntimeError(
                "Model is not fitted."
            )

        return self.content_model.get_similar_movies(
            movie_id,
            top_n=top_n,
        )

    def get_model_info(self):
        """Return hybrid model configuration."""

        return {
            "model": "Hybrid Recommendation System",
            "content_weight": self.content_weight,
            "collaborative_weight": self.collaborative_weight,
            "total_movies": self.content_model.get_movie_count(),
            "total_users": len(
                self.collaborative_model.user_ids
            ),
            "neighbors": self.n_neighbors,
        }


if __name__ == "__main__":

    print("\n" + "=" * 55)
    print("       CINEMATCH HYBRID RECOMMENDER")
    print("=" * 55)

    recommender = HybridRecommender(
        content_weight=0.5,
        collaborative_weight=0.5,
        n_neighbors=10,
    )

    recommender.fit()

    print("\nModel Information:")

    for key, value in recommender.get_model_info().items():
        print(f"{key}: {value}")

    test_user_id = 1

    print(
        f"\nGenerating recommendations for User "
        f"{test_user_id}..."
    )

    try:
        recommendations = recommender.recommend_for_user(
            user_id=test_user_id,
            top_n=TOP_N,
        )

        print("\nFinal Hybrid Recommendations:")

        print(
            recommendations.to_string(index=False)
        )

    except ValueError as error:
        print(f"Recommendation error: {error}")
