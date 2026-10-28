
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity


class CollaborativeModel:
    def __init__(self, ratings):
        self.ratings = ratings.copy()

        self.user_ids = pd_unique(ratings["userId"])
        self.movie_ids = pd_unique(ratings["movieId"])

        self.user_index = {
            user_id: i for i, user_id in enumerate(self.user_ids)
        }
        self.movie_index = {
            movie_id: i for i, movie_id in enumerate(self.movie_ids)
        }

        rows = ratings["userId"].map(self.user_index).to_numpy()
        cols = ratings["movieId"].map(self.movie_index).to_numpy()
        values = ratings["rating"].astype(float).to_numpy()

        self.matrix = csr_matrix(
            (values, (rows, cols)),
            shape=(len(self.user_ids), len(self.movie_ids))
        )

        # Each movie is now represented by the users who rated it.
        self.movie_vectors = normalize(self.matrix.T, norm="l2")

    def get_scores(self, movie_id):
        if movie_id not in self.movie_index:
            return None

        index = self.movie_index[movie_id]

        scores = cosine_similarity(
            self.movie_vectors[index],
            self.movie_vectors
        ).flatten()

        return {
            movie: float(scores[i])
            for i, movie in enumerate(self.movie_ids)
        }


def pd_unique(series):
    return series.drop_duplicates().tolist()
