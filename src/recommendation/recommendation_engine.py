
"""
CineMatch - Recommendation Engine

Central controller that connects recommendation models,
ranking logic and similarity operations.
"""

import pandas as pd

from src.models.hybrid_model import HybridRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.collaborative import CollaborativeRecommender

from src.recommendation.ranking import RecommendationRanker
from src.recommendation.similarity_engine import SimilarityEngine

from config.settings import TOP_N, RECOMMENDATION_THRESHOLD


class RecommendationEngine:
    """Central interface for all CineMatch recommendations."""

    def __init__(
        self,
        content_weight=0.5,
        collaborative_weight=0.5,
        n_neighbors=10,
    ):
        self.hybrid_model = HybridRecommender(
            content_weight=content_weight,
            collaborative_weight=collaborative_weight,
            n_neighbors=n_neighbors,
        )

        self.content_model = ContentBasedRecommender()

        self.collaborative_model = CollaborativeRecommender(
            n_neighbors=n_neighbors
        )

        self.similarity_engine = SimilarityEngine()

        self.ranker = RecommendationRanker()

        self.movies = None
        self.ratings = None

        self.is_fitted = False

    def fit(self, movies=None, ratings=None, features=None, movie_ids=None):
        """Load data and initialize recommendation models."""

        print("\n" + "=" * 55)
        print("       CINEMATCH RECOMMENDATION ENGINE")
        print("=" * 55)

        if movies is None:
            self.hybrid_model.fit(
                features=features,
                movie_ids=movie_ids,
            )

            self.movies = self.hybrid_model.movies.copy()
            self.ratings = self.hybrid_model.ratings.copy()

        else:
            self.movies = movies.copy()

            self.ratings = ratings.copy() if ratings is not None else None

            self.hybrid_model.fit(
                movies=self.movies,
                ratings=self.ratings,
                features=features,
                movie_ids=movie_ids,
            )

        self.content_model = self.hybrid_model.content_model
        self.collaborative_model = self.hybrid_model.collaborative_model

        self.is_fitted = True

        print("\nRecommendation engine initialized successfully!")

        return self

    def _check_ready(self):
        """Ensure models have been initialized."""

        if not self.is_fitted:
            raise RuntimeError(
                "Recommendation engine is not initialized. "
                "Call fit() before requesting recommendations."
            )

    def recommend_for_user(
        self,
        user_id,
        top_n=None,
        diversify=True,
    ):
        """Generate personalized hybrid recommendations."""

        self._check_ready()

        if top_n is None:
            top_n = TOP_N

        if top_n < 1:
            raise ValueError("top_n must be at least 1.")

        recommendations = self.hybrid_model.recommend_for_user(
            user_id=user_id,
            top_n=top_n * 3,
            include_scores=True,
        )

        if recommendations.empty:
            return recommendations

        final_recommendations = self.ranker.get_final_recommendations(
            recommendations,
            top_n=top_n,
            diversify=diversify,
        )

        return final_recommendations

    def recommend_similar_movies(
        self,
        movie_id,
        top_n=None,
    ):
        """Recommend movies similar to a selected movie."""

        self._check_ready()

        if top_n is None:
            top_n = TOP_N

        if top_n < 1:
            raise ValueError("top_n must be at least 1.")

        recommendations = self.content_model.get_similar_movies(
            movie_id=movie_id,
            top_n=top_n,
        )

        return recommendations

    def recommend_by_title(
        self,
        title,
        top_n=None,
    ):
        """Find a movie by title and recommend similar movies."""

        self._check_ready()

        if not isinstance(title, str) or not title.strip():
            raise ValueError("Please provide a valid movie title.")

        matches = self.movies[
            self.movies["title"].str.contains(
                title.strip(),
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

        recommendations = self.recommend_similar_movies(
            movie_id=selected_movie["movie_id"],
            top_n=top_n,
        )

        return recommendations

    def recommend_by_genre(
        self,
        genres,
        top_n=None,
    ):
        """Generate recommendations based on selected genres."""

        self._check_ready()

        if top_n is None:
            top_n = TOP_N

        if top_n < 1:
            raise ValueError("top_n must be at least 1.")

        if isinstance(genres, str):
            genres = [genres]

        selected_genres = [
            genre.lower().strip()
            for genre in genres
            if isinstance(genre, str) and genre.strip()
        ]

        if not selected_genres:
            raise ValueError("At least one valid genre is required.")

        available_genres = [
            genre
            for genre in selected_genres
            if genre in self.movies.columns
        ]

        if not available_genres:
            raise ValueError("No matching movie genres found.")

        genre_mask = self.movies[
            available_genres
        ].eq(1).any(axis=1)

        recommendations = self.movies[
            genre_mask
        ].copy()

        if recommendations.empty:
            return recommendations

        # Use average movie ratings as a basic popularity signal.
        movie_ratings = (
            self.ratings.groupby("movie_id")["rating"]
            .agg(["mean", "count"])
            .reset_index()
        )

        recommendations = recommendations.merge(
            movie_ratings,
            on="movie_id",
            how="left",
        )

        recommendations["mean"] = recommendations["mean"].fillna(0)
        recommendations["count"] = recommendations["count"].fillna(0)

        recommendations = recommendations.sort_values(
            ["mean", "count"],
            ascending=False,
        )

        return recommendations.head(top_n).reset_index(drop=True)

    def get_movie_details(self, movie_id):
        """Return details for a selected movie."""

        self._check_ready()

        result = self.movies[
            self.movies["movie_id"] == int(movie_id)
        ]

        if result.empty:
            raise ValueError(
                f"Movie ID {movie_id} not found."
            )

        return result.iloc[0]

    def get_user_rating_history(self, user_id):
        """Return all ratings submitted by a user."""

        self._check_ready()

        user_id = int(user_id)

        history = self.ratings[
            self.ratings["user_id"] == user_id
        ].copy()

        if history.empty:
            return history

        history = history.merge(
            self.movies[["movie_id", "title"]],
            on="movie_id",
            how="left",
        )

        return history.sort_values(
            "rating",
            ascending=False,
        ).reset_index(drop=True)

    def get_similar_users(self, user_id, top_n=10):
        """Return users with similar rating patterns."""

        self._check_ready()

        return self.collaborative_model.get_similar_users(
            user_id=user_id,
            n_neighbors=top_n,
        )

    def get_movie_similarity(self, movie_id_1, movie_id_2):
        """Calculate similarity between two movies."""

        self._check_ready()

        return self.similarity_engine.get_similarity(
            movie_id_1,
            movie_id_2,
        )

    def get_popular_movies(self, top_n=None, min_ratings=20):
        """Return popular movies using rating count and average."""

        self._check_ready()

        if top_n is None:
            top_n = TOP_N

        if top_n < 1:
            raise ValueError("top_n must be at least 1.")

        popularity = (
            self.ratings.groupby("movie_id")["rating"]
            .agg(["mean", "count"])
            .reset_index()
        )

        popularity = popularity[
            popularity["count"] >= min_ratings
        ]

        popularity = popularity.merge(
            self.movies,
            on="movie_id",
            how="left",
        )

        popularity = popularity.sort_values(
            ["mean", "count"],
            ascending=False,
        )

        return popularity.head(top_n).reset_index(drop=True)

    def get_engine_statistics(self):
        """Return recommendation engine statistics."""

        self._check_ready()

        return {
            "total_movies": len(self.movies),
            "total_users": int(self.ratings["user_id"].nunique()),
            "total_ratings": len(self.ratings),
            "average_rating": round(
                float(self.ratings["rating"].mean()),
                2,
            ),
            "content_weight": self.hybrid_model.content_weight,
            "collaborative_weight": self.hybrid_model.collaborative_weight,
            "recommendation_threshold": RECOMMENDATION_THRESHOLD,
        }


if __name__ == "__main__":

    print("\nInitializing CineMatch Recommendation Engine...")

    engine = RecommendationEngine()

    engine.fit()

    print("\nEngine Statistics:")

    for key, value in engine.get_engine_statistics().items():
        print(f"{key}: {value}")

    test_user_id = 1

    print(f"\nRecommendations for User {test_user_id}:")

    recommendations = engine.recommend_for_user(
        user_id=test_user_id,
        top_n=TOP_N,
    )

    print(recommendations.to_string(index=False))

    print("\nRecommendation engine executed successfully!")
