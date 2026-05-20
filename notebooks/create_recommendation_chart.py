import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "assets" / "screenshots"
OUTPUT_PATH = OUTPUT_DIR / "recommendation-similarity-chart.png"


def load_pickle(path):
    with path.open("rb") as file:
        return pickle.load(file)


def recommend(title, movies, cosine_sim, top_k=10):
    matched = movies.index[movies["title"] == title].tolist()
    if not matched:
        raise ValueError(f"Movie not found: {title}")

    selected_index = matched[0]
    scores = list(enumerate(cosine_sim[selected_index]))
    scores = sorted(scores, key=lambda item: item[1], reverse=True)[1 : top_k + 1]

    movie_indices = [index for index, _ in scores]
    similarities = [score for _, score in scores]
    result = movies.iloc[movie_indices][["title"]].copy()
    result["similarity"] = similarities
    return result


def save_chart(result, selected_title):
    chart_data = result.sort_values("similarity", ascending=True)

    plt.figure(figsize=(11, 7), facecolor="#0b0c10")
    ax = plt.gca()
    ax.set_facecolor("#111217")
    ax.barh(chart_data["title"], chart_data["similarity"], color="#b7333d")

    ax.set_title(
        f"Top 10 Similar Movies for {selected_title}",
        color="#fff6ea",
        fontsize=18,
        fontweight="bold",
        pad=18,
    )
    ax.set_xlabel("Cosine Similarity", color="#d8d0c7", fontsize=12)
    ax.set_ylabel("")
    ax.tick_params(colors="#f5f1ea", labelsize=10)
    ax.grid(axis="x", color="#333640", linewidth=0.8, alpha=0.6)

    for spine in ax.spines.values():
        spine.set_color("#2f323a")

    max_score = max(float(chart_data["similarity"].max()), 0.1)
    ax.set_xlim(0, max_score * 1.12)

    for index, score in enumerate(chart_data["similarity"]):
        ax.text(
            score + max_score * 0.015,
            index,
            f"{score:.3f}",
            va="center",
            color="#fff6ea",
            fontsize=9,
        )

    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=160, bbox_inches="tight", facecolor="#0b0c10")
    plt.close()


def main():
    movies = load_pickle(MODELS_DIR / "movies.pkl")
    cosine_sim = load_pickle(MODELS_DIR / "cosine_sim2.pkl")
    selected_title = "Spider-Man 3"
    result = recommend(selected_title, movies, cosine_sim)
    save_chart(result, selected_title)
    print(f"Saved chart: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
