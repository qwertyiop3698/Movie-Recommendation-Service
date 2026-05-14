# 영화 추천 서비스

TMDB 5000 영화 데이터를 기반으로 만든 Streamlit 콘텐츠 기반 영화 추천 웹앱입니다. 사용자가 영화 한 편을 선택하면 줄거리(`overview`)와 장르(`genres`) 정보를 TF-IDF로 벡터화해 계산한 코사인 유사도 모델을 사용하고, 유사도가 높은 영화 10편을 포스터와 함께 보여줍니다.

앱 실행 시에는 사전 계산된 `cosine_sim2.pkl` 추천 모델을 우선 사용합니다. 이 파일은 줄거리와 장르를 함께 반영한 추천 모델입니다. 로딩에 실패하면 `cosine_sim.pkl`로 자동 대체합니다.

로그인, 회원가입, 사용자 DB, 세션 인증 기능은 사용하지 않습니다.

## 주요 기능

- 영화 제목 선택 후 유사 영화 Top 10 추천
- TMDB 5000 데이터 기반 줄거리 및 장르 정보 정제와 인덱싱
- TF-IDF와 코사인 유사도를 활용한 줄거리/장르 기반 추천
- 5개씩 2줄 그리드 형태의 포스터 카드 UI
- TMDB API를 통한 포스터 이미지 조회 및 한국어 영화 제목 표시
- API 키 누락, 네트워크 오류, 포스터 없음 상황에서 `no_image.png`로 대체
- `movies.pkl`, `cosine_sim2.pkl`, `cosine_sim.pkl` 안전 로딩 및 오류 안내

## 파일 구조

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
    └── Content Based Fitering.ipynb
```

## 설치 방법

Python 가상환경 사용을 권장합니다.

```bash
pip install -r requirements.txt
```

## 환경변수 설정

TMDB API Key는 코드에 직접 넣지 않습니다. 앱은 먼저 Streamlit secrets의 `TMDB_API_KEY`를 확인하고, 없으면 `app.py`와 같은 프로젝트 루트에 있는 `.env` 파일에서 `TMDB_API_KEY`를 읽습니다.

### `.env` 사용

`.env.example`을 참고해 `app.py`와 같은 폴더에 `.env` 파일을 만들고 값을 입력합니다.

```bash
TMDB_API_KEY=발급받은_TMDB_API_KEY
```

`.env` 파일은 Git에 올리지 않습니다.

### Streamlit secrets 사용

`.streamlit/secrets.toml`에 아래처럼 설정할 수도 있습니다.

```toml
TMDB_API_KEY = "발급받은_TMDB_API_KEY"
```

## 실행 방법

```bash
streamlit run app.py
```

브라우저가 열리면 영화 제목을 선택하고 **추천 영화 보기** 버튼을 누르면 됩니다.

## 데이터 파일 안내

앱 실행에는 최소한 아래 파일이 필요합니다.

- `models/movies.pkl`
- `models/cosine_sim2.pkl` 또는 `models/cosine_sim.pkl`
- `assets/no_image.png`

`movies.pkl`에는 최소 `id`, `title` 컬럼이 있어야 합니다. 파일이 없거나 컬럼이 맞지 않으면 웹 화면에 오류 메시지가 표시됩니다.
