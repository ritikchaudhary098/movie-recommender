
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentModel:
    def __init__(self, movies):
        self.movies = movies.reset_index(drop=True)

        self.vectorizer = TfidfVectorizer(stop_words="english")

        self.matrix = self.vectorizer.fit_transform(
            self.movies["features"].fillna("")
        )

    def get_scores(self, movie_id):
        matches = self.movies.index[
            self.movies["movieId"] == movie_id
        ].tolist()

        if not matches:
            raise ValueError(f"Movie ID {movie_id} not found.")

        index = matches[0]

        scores = cosine_similarity(
            self.matrix[index], self.matrix
        ).flatten()

        return scores
