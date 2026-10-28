
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


class MovieAssistant:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        self.model = os.getenv("GROQ_MODEL")

        self.client = (
            Groq(api_key=api_key) if api_key else None
        )

    def explain_recommendations(self, user_request, movies):
        if self.client is None or not self.model:
            return (
                "LLM is not configured. Add GROQ_API_KEY and "
                "GROQ_MODEL to your .env file."
            )

        # Only send a small, relevant set of results to the LLM.
        movie_details = movies[
            ["title", "genres", "avg_rating", "hybrid_score"]
        ].to_dict(orient="records")

        prompt = f"""
The user asked for movie recommendations:
{user_request}

These movies were selected by our Python hybrid recommender:
{movie_details}

Explain in simple language why these results may fit the request.
Mention genres and scores when helpful.
Do not invent movie facts, plots, or ratings.
Do not claim that the recommendations are guaranteed to match.
Keep the answer concise and friendly.
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful movie recommendation assistant. "
                        "Explain only the supplied movie data and be honest "
                        "about uncertainty."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )

        return response.choices[0].message.content
