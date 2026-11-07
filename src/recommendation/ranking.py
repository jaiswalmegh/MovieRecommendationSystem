
"""
CineMatch - Recommendation Ranking Engine

Ranks, merges and filters movie recommendations
from different recommendation sources.
"""

import numpy as np
import pandas as pd

from config.settings import TOP_N


class RecommendationRanker:
    """Ranks and organizes movie recommendations."""

    def __init__(
        self,
        score_weights=None,
        diversity_weight=0.15,
    ):
        self.score_weights = score_weights or {
            "hybrid_score": 0.60,
            "content_score": 0.20,
            "collaborative_score": 0.20,
        }

        self.diversity_weight = diversity_weight

        self._validate_configuration()

    def _validate_configuration(self):
        """Validate ranking configuration."""

        if not 0 <= self.diversity_weight <= 1:
            raise ValueError(
                "Diversity weight must be between 0 and 1."
            )

        for name, weight in self.score_weights.items():
            if weight < 0:
                raise ValueError(
                    f"Weight for {name} cannot be negative."
                )

        if sum(self.score_weights.values()) <= 0:
            raise ValueError(
                "At least one score weight must be positive."
            )

        total = sum(self.score_weights.values())

        self.score_weights = {
            key: value / total
            for key, value in self.score_weights.items()
        }

    @staticmethod
    def normalize_scores(scores):
        """Normalize numeric scores between 0 and 1."""

        values = pd.to_numeric(
            pd.Series(scores),
            errors="coerce",
        )

        minimum = values.min()
        maximum = values.max()

        if pd.isna(minimum) or pd.isna(maximum):
            return pd.Series(
                0.0,
                index=values.index,
            )

        if maximum == minimum:
            return pd.Series(
                1.0,
                index=values.index,
            )

        return (values - minimum) / (maximum - minimum)

    def calculate_ranking_score(self, recommendations):
        """
        Calculate a unified ranking score.

        Available columns are used automatically.
        Missing optional score columns are ignored.
        """

        if recommendations.empty:
            return recommendations.copy()

        if "movie_id" not in recommendations.columns:
            raise ValueError(
                "Recommendations must contain movie_id."
            )

        data = recommendations.copy()

        available_weights = {
            column: weight
            for column, weight in self.score_weights.items()
            if column in data.columns
        }

        if not available_weights:
            if "similarity_score" in data.columns:
                available_weights = {
                    "similarity_score": 1.0
                }
            elif "predicted_rating" in data.columns:
                available_weights = {
                    "predicted_rating": 1.0
                }
            else:
                raise ValueError(
                    "No supported recommendation score column found."
                )

        total_weight = sum(available_weights.values())

        normalized_weights = {
            column: weight / total_weight
            for column, weight in available_weights.items()
        }

        ranking_score = pd.Series(
            0.0,
            index=data.index,
        )

        for column, weight in normalized_weights.items():

            scores = pd.to_numeric(
                data[column],
                errors="coerce",
            )

            # Normalize only when the score is not already
            # a bounded probability-like value.
            if column in (
                "hybrid_score",
                "content_score",
                "collaborative_score",
                "similarity_score",
            ):
                normalized = self.normalize_scores(scores)
            elif column == "predicted_rating":
                normalized = (
                    scores.clip(1, 5) - 1
                ) / 4
            else:
                normalized = self.normalize_scores(scores)

            ranking_score += (
                normalized.fillna(0) * weight
            )

        data["ranking_score"] = ranking_score

        return data

    def merge_recommendations(self, *recommendation_lists):
        """
        Merge multiple recommendation DataFrames.

        Duplicate movie IDs are combined by retaining
        the highest available score for each score column.
        """

        valid_lists = [
            frame.copy()
            for frame in recommendation_lists
            if isinstance(frame, pd.DataFrame)
            and not frame.empty
        ]

        if not valid_lists:
            return pd.DataFrame()

        merged = pd.concat(
            valid_lists,
            ignore_index=True,
            sort=False,
        )

        if "movie_id" not in merged.columns:
            raise ValueError(
                "Every recommendation DataFrame must contain movie_id."
            )

        merged["movie_id"] = pd.to_numeric(
            merged["movie_id"],
            errors="coerce",
        )

        merged = merged.dropna(subset=["movie_id"])

        merged["movie_id"] = merged["movie_id"].astype(int)

        score_columns = [
            "hybrid_score",
            "content_score",
            "collaborative_score",
            "similarity_score",
            "predicted_rating",
        ]

        aggregation = {}

        for column in merged.columns:
            if column == "movie_id":
                continue

            if column in score_columns:
                aggregation[column] = "max"
            else:
                aggregation[column] = "first"

        merged = merged.groupby(
            "movie_id",
            as_index=False,
        ).agg(aggregation)

        return merged

    def rank_recommendations(
        self,
        recommendations,
        top_n=None,
    ):
        """Rank recommendations and return the top results."""

        if top_n is None:
            top_n = TOP_N

        if top_n < 1:
            raise ValueError("top_n must be at least 1.")

        if recommendations is None or recommendations.empty:
            return pd.DataFrame()

        ranked = self.calculate_ranking_score(
            recommendations
        )

        ranked = ranked.sort_values(
            "ranking_score",
            ascending=False,
        )

        ranked = ranked.drop_duplicates(
            subset=["movie_id"],
            keep="first",
        )

        return ranked.head(top_n).reset_index(drop=True)

    def rank_by_rating(
        self,
        recommendations,
        top_n=None,
    ):
        """Rank recommendations using predicted ratings."""

        if top_n is None:
            top_n = TOP_N

        if "predicted_rating" not in recommendations.columns:
            raise ValueError(
                "predicted_rating column is required."
            )

        return (
            recommendations
            .sort_values(
                "predicted_rating",
                ascending=False,
            )
            .drop_duplicates("movie_id")
            .head(top_n)
            .reset_index(drop=True)
        )

    def apply_genre_diversity(
        self,
        recommendations,
        top_n=None,
        max_per_genre=3,
    ):
        """
        Select recommendations while limiting repeated genres.

        Movies without an active genre are still eligible.
        """

        if top_n is None:
            top_n = TOP_N

        if recommendations.empty:
            return recommendations.copy()

        if max_per_genre < 1:
            raise ValueError(
                "max_per_genre must be at least 1."
            )

        genre_columns = [
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

        available_genres = [
            column
            for column in genre_columns
            if column in recommendations.columns
        ]

        if not available_genres:
            return recommendations.head(top_n).reset_index(drop=True)

        ranked = recommendations.sort_values(
            "ranking_score",
            ascending=False,
        )

        genre_counts = {
            genre: 0
            for genre in available_genres
        }

        selected_indices = []

        for index, movie in ranked.iterrows():

            active_genres = [
                genre
                for genre in available_genres
                if movie[genre] == 1
            ]

            if active_genres:
                if any(
                    genre_counts[genre] >= max_per_genre
                    for genre in active_genres
                ):
                    continue

            selected_indices.append(index)

            for genre in active_genres:
                genre_counts[genre] += 1

            if len(selected_indices) >= top_n:
                break

        # Fill remaining slots if genre restrictions leave
        # too few recommendations.
        if len(selected_indices) < top_n:

            remaining = ranked.drop(
                index=selected_indices,
                errors="ignore",
            )

            extra_indices = remaining.head(
                top_n - len(selected_indices)
            ).index.tolist()

            selected_indices.extend(extra_indices)

        return (
            ranked.loc[selected_indices]
            .reset_index(drop=True)
        )

    def get_final_recommendations(
        self,
        *recommendation_lists,
        top_n=None,
        diversify=True,
    ):
        """
        Merge, rank and optionally diversify recommendations.
        """

        if top_n is None:
            top_n = TOP_N

        merged = self.merge_recommendations(
            *recommendation_lists
        )

        if merged.empty:
            return merged

        ranked = self.rank_recommendations(
            merged,
            top_n=len(merged),
        )

        if diversify and "ranking_score" in ranked.columns:
            ranked = self.apply_genre_diversity(
                ranked,
                top_n=top_n,
            )
        else:
            ranked = ranked.head(top_n)

        return ranked.reset_index(drop=True)


if __name__ == "__main__":

    print("\n" + "=" * 55)
    print("       CINEMATCH RECOMMENDATION RANKER")
    print("=" * 55)

    sample_recommendations = pd.DataFrame({
        "movie_id": [1, 2, 3, 4, 5, 1],
        "title": [
            "Toy Story",
            "GoldenEye",
            "Four Rooms",
            "Get Shorty",
            "Copycat",
            "Toy Story",
        ],
        "hybrid_score": [
            0.91, 0.82, 0.75, 0.88, 0.71, 0.91
        ],
        "content_score": [
            0.95, 0.75, 0.70, 0.85, 0.65, 0.95
        ],
        "collaborative_score": [
            0.87, 0.89, 0.80, 0.91, 0.77, 0.87
        ],
        "comedy": [1, 0, 1, 1, 0, 1],
        "action": [0, 1, 0, 0, 0, 0],
        "drama": [0, 0, 0, 0, 1, 0],
    })

    ranker = RecommendationRanker()

    final_results = ranker.get_final_recommendations(
        sample_recommendations,
        top_n=5,
        diversify=True,
    )

    print("\nFinal Ranked Recommendations:\n")

    print(
        final_results[
            ["movie_id", "title", "ranking_score"]
        ].to_string(index=False)
    )

    print("\nRanking engine executed successfully!")
