
from src.data_loader import load_data
from src.content_model import ContentModel
from src.collaborative_model import CollaborativeModel
from src.hybrid_recommender import HybridRecommender


def main():
    movies, ratings = load_data()

    print(f"Movies loaded: {len(movies)}")
    print(f"Ratings loaded: {len(ratings)}")

    content_model = ContentModel(movies)
    collaborative_model = CollaborativeModel(ratings)

    recommender = HybridRecommender(
        movies,
        content_model,
        collaborative_model,
        content_weight=0.4,
    )

    # Select the first available movie for testing.
    selected_movie = movies.iloc[0]
    movie_id = selected_movie["movieId"]

    results = recommender.recommend(movie_id, n=5)

    print(f"\nRecommendations for {selected_movie['title']}:\n")

    print(
        results[
            [
                "title",
                "genres",
                "avg_rating",
                "content_score",
                "collaborative_score",
                "hybrid_score",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
