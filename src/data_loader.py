
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def load_data():
    movies_path = DATA_DIR / "movies.csv"
    ratings_path = DATA_DIR / "ratings.csv"
    tags_path = DATA_DIR / "tags.csv"

    if not movies_path.exists():
        raise FileNotFoundError(f"Missing file: {movies_path}")

    if not ratings_path.exists():
        raise FileNotFoundError(f"Missing file: {ratings_path}")

    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)

    required_movies = {"movieId", "title", "genres"}
    required_ratings = {"userId", "movieId", "rating"}

    if not required_movies.issubset(movies.columns):
        raise ValueError(
            "movies.csv needs movieId, title, and genres columns."
        )

    if not required_ratings.issubset(ratings.columns):
        raise ValueError(
            "ratings.csv needs userId, movieId, and rating columns."
        )

    movies = movies.dropna(subset=["movieId", "title"]).copy()
    movies = movies.drop_duplicates(subset=["movieId"])

    ratings = ratings.dropna(
        subset=["userId", "movieId", "rating"]
    ).copy()

    ratings["rating"] = pd.to_numeric(
        ratings["rating"], errors="coerce"
    )
    ratings = ratings.dropna(subset=["rating"])

    # Calculate average rating and number of ratings per movie.
    rating_summary = (
        ratings.groupby("movieId")["rating"]
        .agg(avg_rating="mean", rating_count="count")
        .reset_index()
    )

    movies = movies.merge(
        rating_summary, on="movieId", how="left"
    )

    movies["avg_rating"] = movies["avg_rating"].fillna(0)
    movies["rating_count"] = (
        movies["rating_count"].fillna(0).astype(int)
    )

    # Read tags if the dataset provides them.
    if tags_path.exists():
        tags = pd.read_csv(tags_path)

        if {"movieId", "tag"}.issubset(tags.columns):
            tags = tags.dropna(subset=["tag"])

            tags_grouped = (
                tags.groupby("movieId")["tag"]
                .agg(lambda values: " ".join(values.astype(str)))
                .reset_index()
            )

            movies = movies.merge(
                tags_grouped, on="movieId", how="left"
            )

    if "tag" not in movies.columns:
        movies["tag"] = ""

    movies["genres"] = movies["genres"].fillna("").astype(str)
    movies["tag"] = movies["tag"].fillna("").astype(str)

    movies["genres"] = movies["genres"].str.replace(
        "|", " ", regex=False
    )

    movies["features"] = (
        movies["genres"] + " " + movies["tag"]
    ).str.lower().str.strip()

    return movies.reset_index(drop=True), ratings
