import base64
from pathlib import Path
from html import escape
import os
import pickle

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from tmdbv3api import Movie, TMDb


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", encoding="utf-8-sig")

MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"

MOVIES_PATH = MODELS_DIR / "movies.pkl"
COSINE_SIM_PATH = MODELS_DIR / "cosine_sim.pkl"
COSINE_SIM2_PATH = MODELS_DIR / "cosine_sim2.pkl"
NO_IMAGE_PATH = ASSETS_DIR / "no_image.png"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"
REQUIRED_MOVIE_COLUMNS = {"id", "title"}


st.set_page_config(
    page_title="영화 추천 서비스",
    page_icon="🎬",
    layout="wide",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                linear-gradient(180deg, rgba(8, 10, 15, 0.96), rgba(16, 14, 18, 0.98)),
                radial-gradient(circle at 18% 12%, rgba(146, 30, 46, 0.28), transparent 28%);
            color: #f5f1ea;
        }

        [data-testid="stHeader"] {
            background: rgba(8, 10, 15, 0.72);
        }

        .block-container {
            padding-top: 3rem;
            padding-bottom: 4rem;
            max-width: 1280px;
        }

        h1, h2, h3, p, label, span {
            letter-spacing: 0;
        }

        .hero-title {
            font-size: 2.9rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
            color: #fff6ea;
        }

        .hero-description {
            max-width: 760px;
            color: #cfc5b8;
            font-size: 1.05rem;
            line-height: 1.7;
            margin-bottom: 1.6rem;
        }

        .section-title {
            color: #fff6ea;
            font-size: 1.35rem;
            font-weight: 700;
            margin: 1.8rem 0 0.8rem;
        }

        div[data-testid="stSelectbox"] label {
            color: #f5f1ea;
            font-weight: 700;
        }

        .stButton > button {
            border: 1px solid rgba(255, 206, 122, 0.35);
            background: linear-gradient(135deg, #b7333d, #6e1e31);
            color: #fff8ed;
            font-weight: 700;
            min-height: 3rem;
            border-radius: 8px;
        }

        .stButton > button:hover {
            border-color: #ffd38a;
            color: #ffffff;
        }

        .movie-card {
            background: rgba(255, 255, 255, 0.055);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            padding: 0.75rem;
            min-height: 100%;
            box-shadow: 0 16px 34px rgba(0, 0, 0, 0.28);
        }

        .movie-poster {
            width: 100%;
            aspect-ratio: 2 / 3;
            object-fit: cover;
            display: block;
            border-radius: 6px;
            border: 1px solid rgba(255, 255, 255, 0.12);
            background: rgba(0, 0, 0, 0.18);
        }

        .movie-title {
            min-height: 3rem;
            margin-top: 0.65rem;
            color: #fff6ea;
            font-weight: 700;
            line-height: 1.35;
            text-align: center;
            overflow-wrap: anywhere;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_pickle(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"{path.name} 파일을 찾을 수 없습니다.")

    with path.open("rb") as file:
        return pickle.load(file)


@st.cache_resource(show_spinner=False)
def load_data():
    movies = load_pickle(MOVIES_PATH)

    missing_columns = REQUIRED_MOVIE_COLUMNS - set(movies.columns)
    if missing_columns:
        columns = ", ".join(sorted(missing_columns))
        raise ValueError(f"movies.pkl에 필요한 컬럼이 없습니다: {columns}")

    try:
        similarity_matrix = load_pickle(COSINE_SIM2_PATH)
        similarity_source = "cosine_sim2.pkl"
    except Exception:
        similarity_matrix = load_pickle(COSINE_SIM_PATH)
        similarity_source = "cosine_sim.pkl"

    if len(similarity_matrix) < len(movies):
        raise ValueError("추천 모델 크기가 영화 목록보다 작습니다. pkl 파일을 확인해 주세요.")

    movies = movies.copy()
    movies["title"] = movies["title"].astype(str)
    return movies, similarity_matrix, similarity_source


def get_tmdb_api_key() -> str:
    try:
        key = st.secrets.get("TMDB_API_KEY")
        if key:
            return str(key)
    except Exception:
        pass

    return os.getenv("TMDB_API_KEY", "")


TMDB_API_KEY = get_tmdb_api_key()


@st.cache_resource(show_spinner=False)
def get_tmdb_client():
    if not TMDB_API_KEY:
        return None

    tmdb = TMDb()
    tmdb.api_key = TMDB_API_KEY
    tmdb.language = "ko-KR"
    return Movie()


@st.cache_data(show_spinner=False)
def get_movie_display_info(movie_id: int, fallback_title: str) -> tuple[str, str]:
    client = get_tmdb_client()
    if client is None:
        return str(NO_IMAGE_PATH), fallback_title

    try:
        movie_detail = client.details(int(movie_id))
        poster_path = movie_detail.get("poster_path") if movie_detail else None
        korean_title = movie_detail.get("title") if movie_detail else None
    except Exception:
        return str(NO_IMAGE_PATH), fallback_title

    poster = f"{TMDB_IMAGE_BASE_URL}{poster_path}" if poster_path else str(NO_IMAGE_PATH)
    title = korean_title or fallback_title
    return poster, title


def get_movie_poster(movie_id: int) -> str:
    poster, _ = get_movie_display_info(movie_id, "")
    return poster


def get_recommendation_movies(title: str, top_k: int = 10):
    movies, similarity_matrix, _ = load_data()
    matched_indices = movies.index[movies["title"] == title].tolist()

    if not matched_indices:
        return [], []

    selected_index = matched_indices[0]
    scores = list(enumerate(similarity_matrix[selected_index]))
    scores.sort(key=lambda item: item[1], reverse=True)

    images = []
    titles = []
    seen_movie_ids = {int(movies.at[selected_index, "id"])}
    seen_titles = {movies.at[selected_index, "title"]}

    for movie_index, _ in scores:
        if movie_index == selected_index or movie_index >= len(movies):
            continue

        movie_row = movies.iloc[movie_index]
        movie_id = int(movie_row["id"])
        movie_title = str(movie_row["title"])

        if movie_id in seen_movie_ids or movie_title in seen_titles:
            continue

        seen_movie_ids.add(movie_id)
        seen_titles.add(movie_title)
        image, display_title = get_movie_display_info(movie_id, movie_title)
        images.append(image)
        titles.append(display_title)

        if len(titles) >= top_k:
            break

    return images, titles


@st.cache_data(show_spinner=False)
def get_image_src(image_path: str) -> str:
    if image_path.startswith("http://") or image_path.startswith("https://"):
        return image_path

    path = Path(image_path)
    if not path.exists():
        return ""

    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render_recommendations(images, titles) -> None:
    if not titles:
        st.info("추천 결과를 찾을 수 없습니다.")
        return

    for row_start in range(0, len(titles), 5):
        cols = st.columns(5)
        for col, image, title in zip(cols, images[row_start : row_start + 5], titles[row_start : row_start + 5]):
            with col:
                image_src = get_image_src(image)
                st.markdown(
                    f"""
                    <div class="movie-card">
                        <img class="movie-poster" src="{escape(image_src)}" alt="{escape(title)} 포스터">
                        <div class="movie-title">{escape(title)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_app() -> None:
    inject_styles()

    st.markdown('<div class="hero-title">영화 추천 서비스</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-description">'
        "마음에 드는 영화 한 편을 고르면, 줄거리와 장르, 출연진 등 콘텐츠 유사도를 바탕으로 "
        "함께 보기 좋은 영화 10편을 추천합니다."
        "</div>",
        unsafe_allow_html=True,
    )

    try:
        movies, _, similarity_source = load_data()
    except Exception as error:
        st.error(f"앱 실행에 필요한 데이터를 불러오지 못했습니다. {error}")
        st.stop()

    if not NO_IMAGE_PATH.exists():
        st.warning("기본 포스터 이미지(no_image.png)를 찾을 수 없습니다.")

    if not TMDB_API_KEY:
        st.warning("TMDB_API_KEY가 설정되지 않아 기본 이미지로 표시됩니다. .env 또는 Streamlit secrets에 키를 등록해 주세요.")

    st.caption(f"추천 모델: {similarity_source}")
    movie_list = np.sort(movies["title"].dropna().unique())
    selected_title = st.selectbox("영화를 선택하세요", movie_list, index=0)

    if st.button("추천 영화 보기", type="primary", use_container_width=True):
        with st.spinner("추천 영화를 찾는 중입니다..."):
            images, titles = get_recommendation_movies(selected_title, top_k=10)

        st.markdown('<div class="section-title">추천 결과</div>', unsafe_allow_html=True)
        render_recommendations(images, titles)


if __name__ == "__main__":
    render_app()
