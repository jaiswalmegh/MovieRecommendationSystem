
"""
CineMatch - Evaluation Metrics

Implements evaluation metrics for movie recommendation systems.
"""

import numpy as np

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
)


class RecommendationMetrics:
    """Calculate recommendation and rating prediction metrics."""

    def __init__(self, relevance_threshold=4.0):
        self.relevance_threshold = relevance_threshold

    @staticmethod
    def _validate_k(k):
        if not isinstance(k, int) or k < 1:
            raise ValueError("K must be a positive integer.")

    def precision_at_k(self, recommended_items, relevant_items, k=10):
        """Calculate Precision@K."""

        self._validate_k(k)

        recommended = list(recommended_items)[:k]
        relevant = set(relevant_items)

        if not recommended:
            return 0.0

        hits = sum(
            1 for item in recommended
            if item in relevant
        )

        return hits / len(recommended)

    def recall_at_k(self, recommended_items, relevant_items, k=10):
        """Calculate Recall@K."""

        self._validate_k(k)

        recommended = list(recommended_items)[:k]
        relevant = set(relevant_items)

        if not relevant:
            return 0.0

        hits = sum(
            1 for item in recommended
            if item in relevant
        )

        return hits / len(relevant)

    def f1_at_k(self, recommended_items, relevant_items, k=10):
        """Calculate F1-Score@K."""

        precision = self.precision_at_k(
            recommended_items,
            relevant_items,
            k,
        )

        recall = self.recall_at_k(
            recommended_items,
            relevant_items,
            k,
        )

        if precision + recall == 0:
            return 0.0

        return (
            2 * precision * recall
            / (precision + recall)
        )

    def hit_rate_at_k(self, recommended_items, relevant_items, k=10):
        """Calculate Hit Rate@K."""

        self._validate_k(k)

        recommended = list(recommended_items)[:k]
        relevant = set(relevant_items)

        return float(
            any(item in relevant for item in recommended)
        )

    def ndcg_at_k(self, recommended_items, relevant_items, k=10):
        """Calculate Normalized Discounted Cumulative Gain@K."""

        self._validate_k(k)

        recommended = list(recommended_items)[:k]
        relevant = set(relevant_items)

        if not relevant:
            return 0.0

        relevance_scores = [
            1.0 if item in relevant else 0.0
            for item in recommended
        ]

        discounts = np.log2(
            np.arange(2, len(relevance_scores) + 2)
        )

        dcg = np.sum(
            np.array(relevance_scores) / discounts
        )

        ideal_length = min(len(relevant), k)

        ideal_relevance = np.ones(ideal_length)

        ideal_discounts = np.log2(
            np.arange(2, ideal_length + 2)
        )

        idcg = np.sum(
            ideal_relevance / ideal_discounts
        )

        if idcg == 0:
            return 0.0

        return float(dcg / idcg)

    def rmse(self, actual_ratings, predicted_ratings):
        """Calculate Root Mean Squared Error."""

        actual = np.asarray(actual_ratings, dtype=float)
        predicted = np.asarray(predicted_ratings, dtype=float)

        if len(actual) == 0 or len(actual) != len(predicted):
            raise ValueError(
                "Actual and predicted ratings must have equal, "
                "non-zero lengths."
            )

        if not np.all(np.isfinite(actual)) or not np.all(
            np.isfinite(predicted)
        ):
            raise ValueError("Ratings must contain finite values.")

        return float(
            np.sqrt(
                mean_squared_error(actual, predicted)
            )
        )

    def mae(self, actual_ratings, predicted_ratings):
        """Calculate Mean Absolute Error."""

        actual = np.asarray(actual_ratings, dtype=float)
        predicted = np.asarray(predicted_ratings, dtype=float)

        if len(actual) == 0 or len(actual) != len(predicted):
            raise ValueError(
                "Actual and predicted ratings must have equal, "
                "non-zero lengths."
            )

        if not np.all(np.isfinite(actual)) or not np.all(
            np.isfinite(predicted)
        ):
            raise ValueError("Ratings must contain finite values.")

        return float(
            mean_absolute_error(actual, predicted)
        )

    def get_relevant_items(self, ratings):
        """Extract relevant movie IDs using the rating threshold."""

        relevant = ratings[
            ratings["rating"] >= self.relevance_threshold
        ]

        return set(relevant["movie_id"].tolist())

    def evaluate_single_user(
        self,
        recommended_items,
        relevant_items,
        k=10,
    ):
        """Calculate all ranking metrics for one user."""

        return {
            f"Precision@{k}": self.precision_at_k(
                recommended_items,
                relevant_items,
                k,
            ),
            f"Recall@{k}": self.recall_at_k(
                recommended_items,
                relevant_items,
                k,
            ),
            f"F1@{k}": self.f1_at_k(
                recommended_items,
                relevant_items,
                k,
            ),
            f"HitRate@{k}": self.hit_rate_at_k(
                recommended_items,
                relevant_items,
                k,
            ),
            f"NDCG@{k}": self.ndcg_at_k(
                recommended_items,
                relevant_items,
                k,
            ),
        }

    def evaluate_multiple_users(
        self,
        user_recommendations,
        user_relevant_items,
        k=10,
    ):
        """Calculate average ranking metrics across multiple users."""

        self._validate_k(k)

        results = []

        for user_id, recommendations in user_recommendations.items():

            relevant = user_relevant_items.get(
                user_id,
                set(),
            )

            metrics = self.evaluate_single_user(
                recommendations,
                relevant,
                k,
            )

            results.append(metrics)

        if not results:
            return {}

        import pandas as pd

        metrics_df = pd.DataFrame(results)

        return metrics_df.mean().to_dict()

    def evaluate_all(
        self,
        actual_ratings,
        predicted_ratings,
    ):
        """Return rating prediction metrics together."""

        return {
            "RMSE": self.rmse(
                actual_ratings,
                predicted_ratings,
            ),
            "MAE": self.mae(
                actual_ratings,
                predicted_ratings,
            ),
        }


if __name__ == "__main__":

    print("\n" + "=" * 55)
    print("       CINEMATCH EVALUATION METRICS")
    print("=" * 55)

    evaluator = RecommendationMetrics(
        relevance_threshold=4.0
    )

    recommended = [101, 102, 103, 104, 105]
    relevant = {101, 103, 106, 108}

    print("\nRanking Evaluation:")

    metrics = evaluator.evaluate_single_user(
        recommended,
        relevant,
        k=5,
    )

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    actual = [4, 5, 3, 2, 5]
    predicted = [4.2, 4.8, 3.5, 2.1, 4.7]

    print("\nRating Prediction Evaluation:")

    rating_metrics = evaluator.evaluate_all(
        actual,
        predicted,
    )

    for name, value in rating_metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nEvaluation metrics executed successfully!")
