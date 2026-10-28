
# Movie Recommendation System

A content-based movie recommendation system built using
Python, Pandas, Scikit-learn, and Streamlit.

## Features

- Search and select a movie.
- Recommend similar movies based on genres and tags.
- Calculate similarity using TF-IDF and cosine similarity.
- Display average ratings and rating counts.
- Interactive web interface using Streamlit.

## Technologies

- Python
- Pandas
- Scikit-learn
- TF-IDF
- Cosine Similarity
- Streamlit
- uv

## Dataset

This project uses MovieLens Latest Small from GroupLens.

Download:
https://grouplens.org/datasets/movielens/latest/

Copy movies.csv, ratings.csv, and tags.csv into:

data/raw/

The links.csv file is included in the dataset but is not
required by the current recommendation model.

## Installation

Install uv if it is not already installed.

Create the environment and install the dependencies:

```bash
uv sync
```

## Run the website

```bash
uv run streamlit run app.py
```

## Test the recommendation model

```bash
uv run python test_recommender.py
```

## How it works

The system combines movie genres and user-generated tags.
TF-IDF converts the combined text into numerical features.
Cosine similarity ranks movies by how similar their features are.

## Future improvements

- Collaborative filtering.
- Hybrid recommendations.
- Recommendation quality evaluation.
- Movie posters and improved interface.
- Online deployment.
