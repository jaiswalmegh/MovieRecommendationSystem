
"""
CineMatch - Model Evaluator

Evaluates and compares:
1. Content-Based Recommendation
2. Collaborative Filtering
3. Hybrid Recommendation

Metrics:
Precision@K, Recall@K, F1@K, HitRate@K, NDCG@K,
RMSE and MAE.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from src.evaluation.metrics import RecommendationMetrics

from config.settings import (
    PROCESSED_DATA_DIR,
    ARTIFACTS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    EVALUATION_K_VALUES,
    RECOMMENDATION_THRESHOLD,
)


class ModelEvaluator:
    """Evaluates and compares movie recommendation models."""

    def __init__(self):

        self.data_dir = Path(PROCESSED_DATA_DIR)
        self.artifacts_dir = Path(ARTIFACTS_DIR)

        self.artifacts_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.metrics = RecommendationMetrics(
            relevance_threshold=RECOMMENDATION_THRESHOLD
        )

        self.results = []

        self.train_ratings = None
        self.test_ratings = None

    # ==========================================
    # DATA LOADING
    # ==========================================

    def load_ratings(self):

        ratings_path = (
            self.data_dir / "ratings_cleaned.csv"
        )

        if not ratings_path.exists():
            raise FileNotFoundError(
                f"Ratings file not found: {ratings_path}"
            )

        return pd.read_csv(ratings_path)

    # ==========================================
    # TRAIN TEST SPLIT
    # ==========================================

    def split_data(self, ratings=None):

        if ratings is None:
            ratings = self.load_ratings()

        ratings = ratings.copy()

        train_parts = []
        test_parts = []

        for _, user_ratings in ratings.groupby("user_id"):

            if len(user_ratings) < 2:
                train_parts.append(user_ratings)
                continue

            train_user, test_user = train_test_split(
                user_ratings,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
            )

            train_parts.append(train_user)
            test_parts.append(test_user)

        if not train_parts or not test_parts:
            raise ValueError(
                "Unable to create train-test split."
            )

        self.train_ratings = pd.concat(
            train_parts,
            ignore_index=True,
        )

        self.test_ratings = pd.concat(
            test_parts,
            ignore_index=True,
        )

        print("\nDataset Split Completed")

        print(
            f"Training ratings: {len(self.train_ratings)}"
        )

        print(
            f"Testing ratings: {len(self.test_ratings)}"
        )

        return self.train_ratings, self.test_ratings

    # ==========================================
    # RECOMMENDATION METHOD ADAPTER
    # ==========================================

    def _get_recommendations(
        self,
        recommender,
        user_id,
        top_n,
    ):
        """
        Support both recommendation method names.

        Hybrid model:
            recommend_for_user()

        Collaborative model:
            get_recommendations()
        """

        if hasattr(recommender, "recommend_for_user"):

            return recommender.recommend_for_user(
                user_id=int(user_id),
                top_n=top_n,
            )

        if hasattr(recommender, "get_recommendations"):

            return recommender.get_recommendations(
                user_id=int(user_id),
                top_n=top_n,
            )

        raise AttributeError(
            f"{recommender.__class__.__name__} "
            "does not have a supported recommendation method."
        )

    # ==========================================
    # RANKING EVALUATION
    # ==========================================

    def evaluate_ranking(
        self,
        recommender,
        model_name,
        k_values=None,
    ):

        if self.train_ratings is None or self.test_ratings is None:
            raise RuntimeError(
                "Run split_data() before evaluation."
            )

        if k_values is None:
            k_values = EVALUATION_K_VALUES

        model_results = []

        test_users = self.test_ratings[
            "user_id"
        ].unique()

        for k in k_values:

            user_metrics = []

            print(
                f"\nEvaluating {model_name} at K={k}..."
            )

            for user_id in test_users:

                relevant_movies = set(
                    self.test_ratings[
                        (self.test_ratings["user_id"] == user_id)
                        & (
                            self.test_ratings["rating"]
                            >= RECOMMENDATION_THRESHOLD
                        )
                    ]["movie_id"].tolist()
                )

                if not relevant_movies:
                    continue

                try:

                    recommendations = self._get_recommendations(
                        recommender,
                        user_id,
                        k,
                    )

                    if recommendations is None or recommendations.empty:
                        recommended_ids = []

                    else:
                        recommended_ids = (
                            recommendations["movie_id"]
                            .astype(int)
                            .tolist()
                        )

                    result = self.metrics.evaluate_single_user(
                        recommended_ids,
                        relevant_movies,
                        k=k,
                    )

                    user_metrics.append(result)

                except (ValueError, KeyError):
                    continue

            if not user_metrics:

                print(
                    f"No evaluable users for {model_name} at K={k}"
                )

                continue

            average_metrics = pd.DataFrame(
                user_metrics
            ).mean()

            result = {
                "model": model_name,
                "K": k,
                "evaluated_users": len(user_metrics),
                **average_metrics.to_dict(),
            }

            model_results.append(result)

        self.results.extend(model_results)

        return pd.DataFrame(model_results)

    # ==========================================
    # RATING PREDICTION EVALUATION
    # ==========================================

    def evaluate_rating_predictions(
        self,
        recommender,
        model_name,
    ):

        if self.test_ratings is None:
            raise RuntimeError(
                "Run split_data() before rating evaluation."
            )

        actual = []
        predicted = []

        for row in self.test_ratings.itertuples(
            index=False
        ):

            try:

                prediction = recommender.predict_rating(
                    int(row.user_id),
                    int(row.movie_id),
                )

                if prediction is None:
                    continue

                if not np.isfinite(prediction):
                    continue

                actual.append(float(row.rating))
                predicted.append(float(prediction))

            except (ValueError, KeyError):
                continue

        if not actual:

            result = {
                "model": model_name,
                "RMSE": np.nan,
                "MAE": np.nan,
                "evaluated_ratings": 0,
            }

        else:

            rating_metrics = self.metrics.evaluate_all(
                actual,
                predicted,
            )

            result = {
                "model": model_name,
                **rating_metrics,
                "evaluated_ratings": len(actual),
            }

        return result

    # ==========================================
    # CONTENT-BASED EVALUATION
    # ==========================================

    def evaluate_content_model(self, model):

        if self.train_ratings is None:
            raise RuntimeError(
                "Training ratings are unavailable."
            )

        class ContentAdapter:

            def __init__(self, content_model, train_ratings):

                self.model = content_model
                self.train_ratings = train_ratings

            def recommend_for_user(
                self,
                user_id,
                top_n=10,
            ):

                user_history = self.train_ratings[
                    self.train_ratings["user_id"] == user_id
                ]

                liked = user_history[
                    user_history["rating"]
                    >= RECOMMENDATION_THRESHOLD
                ]

                if liked.empty:

                    return pd.DataFrame(
                        columns=["movie_id", "title"]
                    )

                watched_ids = set(
                    user_history["movie_id"].astype(int)
                )

                liked_ids = (
                    liked["movie_id"].astype(int).tolist()
                )

                weights = np.maximum(
                    liked["rating"].to_numpy(dtype=float) - 3,
                    0.1,
                )

                candidates = []

                for movie_id in self.model.movie_ids:

                    movie_id = int(movie_id)

                    if movie_id in watched_ids:
                        continue

                    if movie_id not in self.model.movie_id_to_index:
                        continue

                    candidate_index = (
                        self.model.get_movie_index(movie_id)
                    )

                    score = 0.0

                    for liked_id, weight in zip(
                        liked_ids,
                        weights,
                    ):

                        if liked_id not in self.model.movie_id_to_index:
                            continue

                        liked_index = (
                            self.model.get_movie_index(liked_id)
                        )

                        similarity = (
                            self.model.similarity_matrix[
                                liked_index,
                                candidate_index,
                            ]
                        )

                        score += similarity * weight

                    score /= weights.sum()

                    candidates.append({
                        "movie_id": movie_id,
                        "content_score": float(score),
                    })

                if not candidates:

                    return pd.DataFrame(
                        columns=["movie_id", "title"]
                    )

                recommendations = pd.DataFrame(
                    candidates
                )

                recommendations = recommendations.sort_values(
                    "content_score",
                    ascending=False,
                ).head(top_n)

                return recommendations.merge(
                    self.model.movies[
                        ["movie_id", "title"]
                    ],
                    on="movie_id",
                    how="left",
                )

        adapter = ContentAdapter(
            model,
            self.train_ratings,
        )

        return self.evaluate_ranking(
            adapter,
            "Content-Based",
        )

    # ==========================================
    # COLLABORATIVE EVALUATION
    # ==========================================

    def evaluate_collaborative_model(self, model):

        ranking_results = self.evaluate_ranking(
            model,
            "Collaborative",
        )

        rating_results = self.evaluate_rating_predictions(
            model,
            "Collaborative",
        )

        return {
            "ranking": ranking_results,
            "rating_prediction": rating_results,
        }

    # ==========================================
    # HYBRID EVALUATION
    # ==========================================

    def evaluate_hybrid_model(self, model):

        return self.evaluate_ranking(
            model,
            "Hybrid",
        )

    # ==========================================
    # MODEL COMPARISON
    # ==========================================

    def compare_models(self):

        if not self.results:
            raise ValueError(
                "No evaluation results available."
            )

        comparison = pd.DataFrame(
            self.results
        )

        return comparison.sort_values(
            ["K", "model"]
        ).reset_index(drop=True)

    # ==========================================
    # SAVE RESULTS
    # ==========================================

    def save_results(self, results=None):

        if results is None:
            results = self.compare_models()

        output_path = (
            self.artifacts_dir
            / "model_evaluation_results.csv"
        )

        results.to_csv(
            output_path,
            index=False,
        )

        print(
            f"\nEvaluation results saved: {output_path}"
        )

        return output_path


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":

    from src.models.content_based import (
        ContentBasedRecommender
    )

    from src.models.collaborative import (
        CollaborativeRecommender
    )

    from src.models.hybrid_model import (
        HybridRecommender
    )

    from src.data.data_loader import (
        MovieLensDataLoader
    )

    from src.data.data_cleaner import (
        MovieLensDataCleaner
    )

    from src.data.feature_engineering import (
        MovieFeatureEngineer
    )

    print("\n" + "=" * 55)
    print("       CINEMATCH MODEL EVALUATION")
    print("=" * 55)

    evaluator = ModelEvaluator()

    # Step 1: Load raw dataset.
    loader = MovieLensDataLoader()

    raw_data = loader.load_all_data()

    # Step 2: Clean dataset.
    cleaner = MovieLensDataCleaner()

    cleaned_data = cleaner.clean_all_data(
        raw_data
    )

    # Step 3: Split ratings.
    train_ratings, test_ratings = evaluator.split_data(
        cleaned_data["ratings"]
    )

    # Step 4: Generate movie features.
    engineer = MovieFeatureEngineer()

    feature_data = engineer.engineer_features(
        cleaned_data["movies"]
    )

    # Step 5: Train content-based model.
    print("\nTraining Content-Based Model...")

    content_model = ContentBasedRecommender()

    content_model.fit(
        movies=cleaned_data["movies"],
        features=feature_data["features"],
        movie_ids=feature_data["movie_ids"],
    )

    # Step 6: Train collaborative model.
    print("\nTraining Collaborative Model...")

    collaborative_model = CollaborativeRecommender()

    collaborative_model.fit(
        ratings=train_ratings,
        movies=cleaned_data["movies"],
    )

    # Step 7: Train hybrid model.
    print("\nTraining Hybrid Model...")

    hybrid_model = HybridRecommender()

    hybrid_model.fit(
        movies=cleaned_data["movies"],
        ratings=train_ratings,
        features=feature_data["features"],
        movie_ids=feature_data["movie_ids"],
    )

    # Step 8: Evaluate content-based model.
    print("\nEvaluating Content-Based Model...")

    content_results = evaluator.evaluate_content_model(
        content_model
    )

    print(content_results.to_string(index=False))

    # Step 9: Evaluate collaborative model.
    print("\nEvaluating Collaborative Model...")

    collaborative_results = (
        evaluator.evaluate_collaborative_model(
            collaborative_model
        )
    )

    print(
        collaborative_results["ranking"].to_string(
            index=False
        )
    )

    print("\nCollaborative Rating Prediction:")

    print(
        collaborative_results["rating_prediction"]
    )

    # Step 10: Evaluate hybrid model.
    print("\nEvaluating Hybrid Model...")

    hybrid_results = evaluator.evaluate_hybrid_model(
        hybrid_model
    )

    print(hybrid_results.to_string(index=False))

    # Step 11: Compare all models.
    print("\n" + "=" * 55)
    print("       FINAL MODEL COMPARISON")
    print("=" * 55)

    comparison = evaluator.compare_models()

    print(
        comparison.to_string(index=False)
    )

    # Step 12: Save evaluation results.
    evaluator.save_results(
        comparison
    )

    print("\n" + "=" * 55)
    print("       EVALUATION COMPLETED")
    print("=" * 55)
