import streamlit as st
import plotly.express as px

from src.data_loader import load_data
from src.content_model import ContentModel
from src.collaborative_model import CollaborativeModel
from src.hybrid_recommender import HybridRecommender
from src.llm_assistant import MovieAssistant


# 1. Page configuration
st.set_page_config(
    page_title="AI Movie Recommender",
    page_icon="🎬",
    layout="wide",
)


# 2. Load data and recommendation models
@st.cache_resource
def load_system():
    movies, ratings = load_data()

    content_model = ContentModel(movies)
    collaborative_model = CollaborativeModel(ratings)

    recommender = HybridRecommender(
        movies,
        content_model,
        collaborative_model,
        content_weight=0.4,
    )

    return movies, ratings, recommender


try:
    movies, ratings, recommender = load_system()
except Exception as error:
    st.error(f"Could not load the recommendation system: {error}")
    st.stop()


# 3. Main page
st.title("🎬 AI Movie Recommendation System")
st.write(
    "Discover movies using hybrid machine learning "
    "and get explanations from an AI assistant."
)

recommend_tab, analytics_tab = st.tabs(
    ["🎥 Movie Recommendations", "📊 Movie Analytics"]
)


# 4. Movie recommendation tab
with recommend_tab:
    st.subheader("Find Your Next Movie")

    titles = sorted(movies["title"].dropna().unique())

    selected_title = st.selectbox(
        "Choose a movie you like",
        titles,
    )

    number_to_show = st.slider(
        "Number of recommendations",
        min_value=3,
        max_value=15,
        value=5,
    )

    user_request = st.text_input(
        "What kind of movies do you want?",
        placeholder="Example: Romantic movies with an emotional story",
    )

    if st.button("Recommend Movies", type="primary"):
        selected_rows = movies[movies["title"] == selected_title]

        if selected_rows.empty:
            st.warning("Movie not found.")
        else:
            selected_id = selected_rows.iloc[0]["movieId"]

            try:
                results = recommender.recommend(
                    selected_id,
                    n=number_to_show,
                )

                st.session_state["last_results"] = results
                st.session_state["last_request"] = (
                    user_request
                    or f"Movies similar to {selected_title}"
                )

                st.subheader(f"Movies related to {selected_title}")

                if results.empty:
                    st.info("No recommendations were found.")
                else:
                    # Display all recommendation scores
                    display_columns = [
                        "title",
                        "genres",
                        "avg_rating",
                        "rating_count",
                        "content_score",
                        "collaborative_score",
                        "hybrid_score",
                    ]

                    missing_columns = [
                        column
                        for column in display_columns
                        if column not in results.columns
                    ]

                    if missing_columns:
                        st.error(
                            f"Missing columns: {missing_columns}. "
                            "Check src/hybrid_recommender.py."
                        )
                    else:
                        st.dataframe(
                            results[display_columns].round(3),
                            use_container_width=True,
                            hide_index=True,
                        )

                        st.caption(
                            "Scores are ranking values, not probabilities "
                            "that you will like a movie."
                        )

                        with st.expander("Understand the scores"):
                            st.markdown(
                                """
                                - **Content score:** Similarity based on
                                  movie genres and tags.
                                - **Collaborative score:** Similarity based
                                  on users' rating patterns.
                                - **Hybrid score:** Combines both scores
                                  with a small rating-based adjustment.
                                """
                            )

            except Exception as error:
                st.error(f"Recommendation failed: {error}")


# 5. AI explanation
with recommend_tab:
    if st.button("✨ Explain these recommendations with AI"):
        results = st.session_state.get("last_results")
        request = st.session_state.get("last_request", "")

        if results is None:
            st.warning("Please generate recommendations first.")
        elif results.empty:
            st.warning("There are no recommendations to explain.")
        else:
            with st.spinner("AI is explaining your recommendations..."):
                try:
                    assistant = MovieAssistant()

                    explanation = assistant.explain_recommendations(
                        request,
                        results,
                    )

                    st.subheader("🤖 AI Explanation")
                    st.markdown(explanation)

                except Exception as error:
                    st.error(
                        "AI explanation failed. Check your Groq API key, "
                        "model name, internet connection, and usage limits. "
                        f"Details: {error}"
                    )


# 6. Movie analytics tab
with analytics_tab:
    st.subheader("Explore the Movie Dataset")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Movies", f"{len(movies):,}")
    col2.metric("Total Ratings", f"{len(ratings):,}")
    col3.metric("Unique Users", f"{ratings['userId'].nunique():,}")

    st.divider()

    # Movies by genre
    st.subheader("Movies by Genre")

    genre_counts = (
        movies.assign(
            genre=movies["genres"].fillna("").str.split("|", regex=False)
        )
        .explode("genre")
    )

    genre_counts["genre"] = genre_counts["genre"].str.strip()

    genre_counts = genre_counts[
        (genre_counts["genre"] != "")
        & (genre_counts["genre"] != "(no genres listed)")
    ]

    genre_counts = (
        genre_counts.groupby("genre")
        .size()
        .reset_index(name="movie_count")
        .sort_values("movie_count", ascending=False)
        .head(15)
    )

    if genre_counts.empty:
        st.info("No genre information was found.")
    else:
        genre_chart = px.bar(
            genre_counts,
            x="genre",
            y="movie_count",
            title="Top 15 Genres by Movie Count",
            labels={
                "genre": "Genre",
                "movie_count": "Number of Movies",
            },
        )

        st.plotly_chart(genre_chart, use_container_width=True)

    st.divider()

    # Top-rated movies
    st.subheader("⭐ Top-Rated Movies")

    minimum_ratings = st.slider(
        "Minimum number of ratings",
        min_value=1,
        max_value=100,
        value=5,
        key="minimum_ratings",
    )

    top_rated = (
        movies[movies["rating_count"] >= minimum_ratings]
        .sort_values(
            ["avg_rating", "rating_count"],
            ascending=[False, False],
        )
        .head(10)
    )

    if top_rated.empty:
        st.info("No movies meet this rating-count requirement.")
    else:
        st.dataframe(
            top_rated[
                ["title", "genres", "avg_rating", "rating_count"]
            ].round(2),
            use_container_width=True,
            hide_index=True,
        )