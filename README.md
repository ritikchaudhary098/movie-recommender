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
```

The system normalizes similarity scores, applies a small average-rating adjustment, excludes the selected movie, and returns the top-ranked recommendations.

### 5. AI-Powered Explanations

- Integrates a Groq-hosted language model.
- Generates natural-language explanations for recommendations.
- Uses movie information and recommendation scores as context.

**Note:** The current LLM integration explains recommendations; it does not directly change their ranking based on the user's text preferences.

### 6. Movie Analytics Dashboard

- Displays total movies, ratings, and unique users.
- Visualizes genre distribution.
- Shows highly rated movies.
- Provides an interactive interface using Streamlit.

## 📁 Project Structure

```text
movie-recommender/
│
├── app.py
├── test_recommender.py
├── pyproject.toml
├── uv.lock
├── .env.example
├── .gitignore
│
├── data/
│   └── raw/
│       ├── movies.csv
│       ├── ratings.csv
│       └── tags.csv
│
└── src/
    ├── __init__.py
    ├── data_loader.py
    ├── content_model.py
    ├── collaborative_model.py
    ├── hybrid_recommender.py
    └── llm_assistant.py
```

## 📊 Dataset

The project expects the following CSV files inside `data/raw/`:

| File | Description |
|---|---|
| `movies.csv` | Movie IDs, titles, and genres |
| `ratings.csv` | User IDs, movie IDs, and ratings |
| `tags.csv` | Optional user-generated movie tags |

**Required columns:**

- `movies.csv`: `movieId`, `title`, `genres`
- `ratings.csv`: `userId`, `movieId`, `rating`
- `tags.csv`: `userId`, `movieId`, `tag`

Make sure your dataset files use the expected column names.

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.12 or a compatible version
- uv package manager
- Groq API key for AI-generated explanations

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/movie-recommender.git
cd movie-recommender
```

Replace `YOUR_USERNAME` with your GitHub username and use your actual repository URL.

### 2. Install Dependencies

```bash
uv sync
```

If dependencies are not already configured, add them with:

```bash
uv add pandas numpy scikit-learn scipy streamlit plotly groq python-dotenv
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b
```

Replace the placeholder with your actual API key and verify the model is available to your account.

**Important:** Never upload your real `.env` file or API key to GitHub.

### 4. Add the Dataset

Place your CSV files in:

```text
data/raw/movies.csv
data/raw/ratings.csv
data/raw/tags.csv
```

### 5. Run the Application

```bash
uv run streamlit run app.py
```

Open the local URL displayed in your terminal to access the application.

## 🧪 Run Tests

```bash
uv run python test_recommender.py
```

## ⚠️ Limitations

- Recommendation quality depends on the available dataset.
- New movies with no rating history may receive limited collaborative recommendations.
- Sparse user-rating data can reduce similarity reliability.
- Hybrid weights have not necessarily been optimized.
- LLM explanations do not currently control recommendation ranking.

## 🔮 Future Improvements

- Evaluate recommendations using Precision@K, Recall@K, and NDCG@K.
- Tune hybrid model weights.
- Improve collaborative filtering with rating normalization.
- Add genre, year, and rating filters.
- Personalize recommendations using user preferences.
- Use natural-language preferences to influence ranking.
- Deploy the application online.

## 💡 Key Skills Demonstrated

Python • Data Preprocessing • Exploratory Data Analysis • Feature Engineering • TF-IDF • Cosine Similarity • Collaborative Filtering • Hybrid Recommendation Systems • Machine Learning • Streamlit • Data Visualization • Generative AI • LLM API Integration • Git

## 👨‍💻 Author

**Ritik Chaudhary**

B.Tech Computer Science Engineering — Artificial Intelligence and Machine Learning

- GitHub: https://github.com/ritikchaudhari098
- LinkedIn: Add your LinkedIn profile URL

---

⭐ If you find this project useful, consider giving the repository a star!
