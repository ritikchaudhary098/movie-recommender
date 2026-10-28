
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommender:

    def __init__(self, movies: pd.DataFrame):
        # Store the movie dataset
        self.movies = movies.reset_index(drop=True)

        # Convert movie text into numerical features
        self.vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.movies["features"].fillna("")
        )

        # Map each movie title to its row index
        self.title_to_index = {
            title: index
            for index, title in enumerate(self.movies["title"])
        }

    def recommend(self, movie_title: str, n: int = 5):
        """Return the top n movies similar to the selected movie."""

        # Validate the number of recommendations
        if n < 1:
            raise ValueError(
                "The number of recommendations must be at least 1."
            )

        # Check whether the movie exists
        if movie_title not in self.title_to_index:
            raise ValueError(
                f"Movie '{movie_title}' was not found."
            )

        # Find the selected movie's index
        movie_index = self.title_to_index[movie_title]

        # Compare the selected movie with all movies
        similarity_scores = cosine_similarity(
            self.tfidf_matrix[movie_index],
            self.tfidf_matrix,
        ).flatten()

        # Sort indices from highest similarity to lowest
        ranked_indices = similarity_scores.argsort()[::-1]

        recommendations = []

        # Collect the most similar movies
        for index in ranked_indices:

            # Skip the selected movie itself
            if index == movie_index:
                continue

            # Copy the movie information
            movie = self.movies.iloc[index].copy()

            # Add its similarity score
            movie["similarity"] = float(
                similarity_scores[index]
            )

            recommendations.append(movie)

            # Stop after finding n recommendations
            if len(recommendations) >= n:
                break

        # Return results as a DataFrame
        if not recommendations:
            return pd.DataFrame()

        return pd.DataFrame(recommendations).reset_index(drop=True)
