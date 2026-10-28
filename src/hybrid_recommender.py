
import numpy as np


class HybridRecommender:
    def __init__(
        self,
        movies,
        content_model,
        collaborative_model,
        content_weight=0.4,
    ):
        self.movies = movies.reset_index(drop=True)
        self.content_model = content_model
        self.collaborative_model = collaborative_model
        self.content_weight = content_weight

    def recommend(self, movie_id, n=10):
        if n < 1:
            raise ValueError("n must be at least 1.")

        content_scores = self.content_model.get_scores(movie_id)

        collaborative_scores = (
            self.collaborative_model.get_scores(movie_id)
        )

        results = self.movies.copy()
        results["content_score"] = content_scores

        if collaborative_scores is None:
            results["collaborative_score"] = 0.0
        else:
            results["collaborative_score"] = (
                results["movieId"]
                .map(collaborative_scores)
                .fillna(0.0)
            )

        # Normalize each score so the two models are comparable.
        for column in ["content_score", "collaborative_score"]:
            maximum = results[column].max()
            if maximum > 0:
                results[column] = results[column] / maximum

        results["hybrid_score"] = (
            self.content_weight * results["content_score"]
            + (1 - self.content_weight)
            * results["collaborative_score"]
        )

        # Use a small rating-based adjustment for tie-breaking.
        results["hybrid_score"] += (
            0.02 * results["avg_rating"].clip(0, 5) / 5
        )

        # Don't recommend the movie the user selected.
        results = results[
            results["movieId"] != movie_id
        ]

        results = results.sort_values(
            "hybrid_score", ascending=False
        )

        return results.head(n).reset_index(drop=True)
