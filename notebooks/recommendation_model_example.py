import ast
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"


def parse_names(value):
    items = ast.literal_eval(value) if isinstance(value, str) else value
    if not isinstance(items, list):
        return []
    return [item["name"].replace(" ", "").lower() for item in items if "name" in item]


def create_recommendation_text(row):
    genre_text = " ".join(row["genres"])
    return f"{row['overview']} {(genre_text + ' ') * 3}".strip()


def build_model():
    movies_df = pd.read_csv(DATA_DIR / "tmdb_5000_movies.csv")
    credits_df = pd.read_csv(DATA_DIR / "tmdb_5000_credits.csv")

    df = pd.merge(
        movies_df,
        credits_df[["movie_id", "cast", "crew"]],
        left_on="id",
        right_on="movie_id",
        how="left",
    )

    df["overview"] = df["overview"].fillna("")
    df["genres"] = df["genres"].apply(parse_names)
    df["recommendation_text"] = df.apply(create_recommendation_text, axis=1)

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df["recommendation_text"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    model_movies = df[["id", "title"]].copy()

    MODELS_DIR.mkdir(exist_ok=True)
    with (MODELS_DIR / "movies.pkl").open("wb") as file:
        pickle.dump(model_movies, file)
    with (MODELS_DIR / "cosine_sim2.pkl").open("wb") as file:
        pickle.dump(cosine_sim, file)

    return model_movies, cosine_sim


def recommend(title, movies, cosine_sim, top_k=10):
    matched = movies.index[movies["title"] == title].tolist()
    if not matched:
        return pd.DataFrame(columns=["title", "similarity"])

    selected_index = matched[0]
    scores = list(enumerate(cosine_sim[selected_index]))
    scores = sorted(scores, key=lambda item: item[1], reverse=True)[1 : top_k + 1]

    movie_indices = [index for index, _ in scores]
    similarities = [score for _, score in scores]

    result = movies.iloc[movie_indices][["title"]].copy()
    result["similarity"] = similarities
    return result


def plot_recommendations(result, title):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=result, x="similarity", y="title", color="#b7333d")
    plt.title(f"Top Recommendations for {title}")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Movie Title")
    plt.xlim(0, max(result["similarity"].max() * 1.1, 0.1))
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    movies, cosine_sim2 = build_model()
    selected_title = "Spider-Man 3"
    recommendations = recommend(selected_title, movies, cosine_sim2)
    print(recommendations)
    plot_recommendations(recommendations, selected_title)
