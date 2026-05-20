# 영화 추천 웹앱

Streamlit으로 만든 콘텐츠 기반 영화 추천 웹앱입니다. 사용자가 영화 제목을 선택하면, 해당 영화와 줄거리 및 장르 특성이 비슷한 영화 10편을 추천합니다.

이 프로젝트는 사용자의 평점이나 시청 기록을 사용하지 않습니다. TMDB 5000 영화 데이터셋에 포함된 영화의 `overview`와 `genres` 정보를 바탕으로 TF-IDF 벡터를 만들고, 코사인 유사도를 계산해 가까운 영화를 찾습니다.

## 주요 기능

- 영화 제목 선택 기반 추천
- 선택한 영화와 유사한 영화 Top 10 표시
- TF-IDF와 코사인 유사도를 활용한 콘텐츠 기반 필터링
- TMDB API를 통한 포스터와 한국어 제목 표시
- API 키가 없거나 포스터가 없을 때 기본 이미지로 대체
- `cosine_sim2.pkl` 로딩 실패 시 `cosine_sim.pkl`로 대체

## 추천 방식

추천 모델은 영화마다 추천 문장을 만든 뒤, 영화 간 유사도를 계산합니다.

1. 영화 데이터에서 줄거리 `overview`를 가져옵니다.
2. 장르 `genres` 값을 파싱해 추천 문장에 함께 넣습니다.
3. 장르 특성이 추천 결과에 더 잘 반영되도록 장르 텍스트를 3회 반복합니다.
4. 추천 문장을 `TfidfVectorizer`로 벡터화합니다.
5. 영화 벡터 간 `cosine_similarity`를 계산합니다.
6. 사용자가 선택한 영화 자신은 제외하고, 유사도가 높은 영화 10편을 반환합니다.

추천 문장은 다음과 같은 방식으로 구성됩니다.

```python
def create_recommendation_text(row):
    genre_text = " ".join(row["genres"])
    return f"{row['overview']} {(genre_text + ' ') * 3}".strip()
```

## 기술 스택

- Python
- Streamlit
- pandas
- numpy
- scikit-learn
- tmdbv3api
- python-dotenv
- matplotlib
- seaborn

## 프로젝트 구조

```text
.
├── app.py
├── requirements.txt
├── .env.example
├── README.md
├── assets/
│   └── no_image.png
├── data/
│   ├── tmdb_5000_movies.csv
│   └── tmdb_5000_credits.csv
├── models/
│   ├── movies.pkl
│   ├── cosine_sim.pkl
│   └── cosine_sim2.pkl
└── notebooks/
    ├── Content Based Fitering.ipynb
    ├── create_recommendation_chart.py
    └── recommendation_model_example.py
```

## 설치 및 실행

먼저 필요한 패키지를 설치합니다.

```bash
pip install -r requirements.txt
```

그다음 Streamlit 앱을 실행합니다.

```bash
streamlit run app.py
```

실행 후 브라우저에서 Streamlit이 안내하는 로컬 주소로 접속하면 앱을 사용할 수 있습니다.

## 환경 변수 설정

TMDB 포스터와 한국어 제목 정보를 사용하려면 프로젝트 루트에 `.env` 파일을 만들고 API 키를 설정합니다.

```bash
TMDB_API_KEY=발급받은_TMDB_API_KEY
```

`.env.example` 파일을 참고해 같은 형식으로 작성하면 됩니다.

TMDB API 키가 없어도 추천 기능 자체는 동작합니다. 다만 포스터와 TMDB 기반 제목 정보는 표시되지 않고 기본 이미지가 사용됩니다.

## 필수 파일

앱 실행을 위해 다음 파일이 필요합니다.

- `models/movies.pkl`
- `models/cosine_sim2.pkl` 또는 `models/cosine_sim.pkl`
- `assets/no_image.png`

`movies.pkl`에는 최소한 `id`, `title` 컬럼이 포함되어 있어야 합니다.

## 모델 파일 재생성

추천 모델 파일은 `notebooks/recommendation_model_example.py`에서 재생성할 수 있습니다.

```bash
python notebooks/recommendation_model_example.py
```

이 스크립트는 `data/tmdb_5000_movies.csv`와 `data/tmdb_5000_credits.csv`를 읽고 다음 파일을 생성합니다.

- `models/movies.pkl`
- `models/cosine_sim2.pkl`

모델 생성 과정의 핵심 코드는 다음과 같습니다.

```python
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["recommendation_text"])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
```

## 추천 함수 예시

```python
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
```

## 데이터

이 프로젝트는 TMDB 5000 영화 데이터셋을 사용합니다.

- `data/tmdb_5000_movies.csv`
- `data/tmdb_5000_credits.csv`

현재 추천 로직에서는 주로 영화 줄거리와 장르 정보를 사용합니다. `credits` 데이터는 모델 예제 스크립트에서 병합되지만, 현재 추천 문장에는 배우나 감독 정보가 포함되어 있지 않습니다.

## 동작 흐름

1. 앱 시작 시 `models/movies.pkl`과 유사도 행렬 파일을 로드합니다.
2. 사용자가 영화 제목을 선택합니다.
3. 선택한 영화의 유사도 점수를 기준으로 추천 후보를 정렬합니다.
4. 자기 자신과 중복 영화를 제외합니다.
5. 상위 10개 영화의 포스터와 제목을 화면에 표시합니다.

## 참고 사항

- `cosine_sim2.pkl`은 줄거리와 장르 가중치를 함께 사용한 개선 모델입니다.
- `cosine_sim.pkl`은 대체용 기본 유사도 행렬로 사용됩니다.
- TMDB API 호출이 실패하면 앱은 기본 이미지와 원본 영화 제목을 사용합니다.
- `.env` 파일에는 개인 API 키가 들어갈 수 있으므로 Git에 포함하지 않는 것이 좋습니다.
