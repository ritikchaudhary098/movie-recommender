# 🎬 Movie Recommendation System

A hybrid movie recommendation system built using Python, Machine Learning, and Generative AI. It recommends movies based on genres, tags, and user rating patterns, with AI-generated explanations powered by the Groq API.

## 🚀 Features

- 🎥 Content-Based Filtering using TF-IDF and cosine similarity
- 👥 Collaborative Filtering using user–movie rating patterns
- 🔀 Hybrid Recommendation Engine combining both approaches
- 🤖 AI-generated recommendation explanations using Groq LLM
- 📊 Interactive movie analytics dashboard using Streamlit
- ⭐ Movie ratings and rating-count analysis
- 🧪 Automated tests for recommendation functionality

## 🛠️ Tech Stack

- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Sparse Matrices:** SciPy
- **Web Application:** Streamlit
- **Visualizations:** Plotly
- **Generative AI:** Groq API
- **Environment Management:** python-dotenv
- **Package Manager:** uv
- **Version Control:** Git and GitHub

## 🧠 How It Works

### 1. Data Preprocessing
- Loads movie, rating, and tag datasets from CSV files.
- Handles missing values and duplicate records.
- Calculates average movie ratings and rating counts.
- Combines movie genres and tags into text features.

### 2. Content-Based Filtering
- Uses movie genres and tags to identify similar movies.
- Converts text features into numerical vectors using `TfidfVectorizer`.
- Calculates similarity using cosine similarity.
- Recommends movies with similar characteristics.

### 3. Collaborative Filtering
- Builds a user–movie rating matrix.
- Represents movies using user-rating patterns.
- Uses cosine similarity to identify movies with similar rating patterns.
- Recommends movies based on relationships in user ratings.

### 4. Hybrid Recommendation Engine

Combines content-based and collaborative similarity scores into a single ranking.

**Hybrid Score:**

```text
Hybrid Score = 0.4 × Content Score
             + 0.6 × Collaborative Score
             + Rating Bonus
