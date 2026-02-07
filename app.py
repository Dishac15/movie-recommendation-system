import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="centered"
)

st.markdown(
    """
    <style>
    /* Main background */
    .stApp {
        background: linear-gradient(180deg, #020617, #020617);
        color: #e5e7eb;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Title */
    h1 {
        color: #f8fafc;
        text-align: center;
        letter-spacing: 1px;
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #9ca3af;
        font-size: 18px;
        margin-bottom: 30px;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #4f46e5);
        color: white;
        border-radius: 12px;
        padding: 0.6em 1.8em;
        font-size: 16px;
        border: none;
        box-shadow: 0px 8px 20px rgba(79, 70, 229, 0.4);
        transition: 0.3s ease;
    }

    .stButton > button:hover {
        transform: scale(1.05);
        background: linear-gradient(135deg, #8b5cf6, #6366f1);
    }

    /* Selectbox */
    .stSelectbox div {
        background-color: #020617 !important;
        color: white;
        border-radius: 10px;
    }

    /* Movie card */
    .movie-card {
        background: #020617;
        border-radius: 15px;
        padding: 15px;
        margin-bottom: 20px;
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.6);
        transition: 0.3s ease;
    }

    .movie-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 25px 60px rgba(124, 58, 237, 0.4);
    }

    .movie-title {
        text-align: center;
        font-size: 18px;
        font-weight: 600;
        margin-top: 10px;
    }

    .movie-meta {
        text-align: center;
        font-size: 14px;
        color: #c7d2fe;
        margin-bottom: 8px;
    }

    .movie-plot {
        font-size: 13px;
        color: #9ca3af;
        line-height: 1.4;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Load data
movies = pd.read_csv("dataset/tmdb_5000_movies.csv", encoding="latin-1")
credits = pd.read_csv("dataset/tmdb_5000_credits.csv", encoding="latin-1")

# Merge datasets
movies = movies.merge(credits, on="title")

# Select required columns
movies = movies[["movie_id", "title", "overview", "genres", "keywords"]]
movies.dropna(inplace=True)

# Convert text columns to string
movies["overview"] = movies["overview"].astype(str)
movies["genres"] = movies["genres"].astype(str)
movies["keywords"] = movies["keywords"].astype(str)

# Create tags
movies["tags"] = movies["overview"] + " " + movies["genres"] + " " + movies["keywords"]

# Vectorization
cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(movies["tags"]).toarray()
similarity = cosine_similarity(vectors)

# Recommendation function
def recommend(movie):
    index = movies[movies["title"] == movie].index[0]
    distances = sorted(
        list(enumerate(similarity[index])),
        reverse=True,
        key=lambda x: x[1]
    )
    recommended_movies = []
    for i in distances[1:6]:
        recommended_movies.append(movies.iloc[i[0]].title)
    return recommended_movies

OMDB_API_KEY = "e6cb1ff0"

def fetch_movie_details(movie_title):
    url = "http://www.omdbapi.com/"
    params = {
        "t": movie_title,
        "apikey": OMDB_API_KEY
    }

    response = requests.get(url, params=params)
    data = response.json()

    if data.get("Response") == "True":
        return {
            "poster": data.get("Poster"),
            "year": data.get("Year"),
            "rating": data.get("imdbRating"),
            "plot": data.get("Plot")
        }
    else:
        return None


# Streamlit UI
st.markdown(
    """
    <h1>🎬 Movie Recommendation System</h1>
    <div class="subtitle">
        Discover movies with the same vibe, story, and energy
    </div>
    """,
    unsafe_allow_html=True
)



st.markdown("### 🎥 Select a Movie")

selected_movie = st.selectbox(
    "",
    sorted(movies['title'].values)
)


if st.button("🎯 Recommend"):
    recommendations = recommend(selected_movie)

    st.markdown("## 🍿 Recommended Movies")

    cols = st.columns(3)

    for idx, movie in enumerate(recommendations):
        details = fetch_movie_details(movie)

        with cols[idx % 3]:
            st.markdown("<div class='movie-card'>", unsafe_allow_html=True)

            if details and details["poster"] != "N/A":
                st.image(details["poster"], use_container_width=True)

            st.markdown(
                f"""
                <div class="movie-title">{movie}</div>
                <div class="movie-meta">
                    📅 {details['year']} &nbsp; ⭐ {details['rating']}
                </div>
                <div class="movie-plot">
                    {details['plot']}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("</div>", unsafe_allow_html=True)
