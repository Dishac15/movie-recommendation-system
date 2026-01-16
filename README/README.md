# Movie Recommendation System

## Overview
This project implements a **content-based movie recommendation system** using natural language processing techniques. It recommends movies similar to a given movie based on textual features like plot overviews, genres, and keywords. The system leverages TF-IDF vectorization and cosine similarity to compute similarity scores between movies.

The model processes a dataset of 5000 movies from The Movie Database (TMDb) and generates recommendations by combining descriptive tags (overview + genres + keywords) for each movie.

### Key Features
- **Content-Based Filtering**: Recommends movies with similar content profiles.
- **Simple and Efficient**: Uses scikit-learn for vectorization and similarity computation.
- **Interactive**: Run in a Jupyter notebook for easy experimentation.

## Dataset
- **Source**: TMDb 5000 Movies Dataset (available as `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv`).
- **Key Columns Used**:
  - `movie_id`: Unique identifier.
  - `title`: Movie title.
  - `overview`: Plot summary.
  - `genres`: List of genres (e.g., ["Action", "Adventure"]).
  - `keywords`: List of keywords (e.g., ["culture clash", "future war"]).
- **Preprocessing**: Merges movies and credits data, extracts names from list-like columns, handles missing values.

## Approach
1. **Data Loading & Merging**: Load CSV files, merge on `title`, select relevant columns, drop NaNs.
2. **Feature Extraction**:
   - Convert genres and keywords from JSON-like strings to clean text (e.g., "Action Adventure").
   - Create a `tags` column: Concatenate `overview + genres + keywords`.
3. **Vectorization**: Use `CountVectorizer` (TF-IDF style with max 5000 features, English stop words) to transform tags into a sparse matrix.
4. **Similarity Computation**: Calculate cosine similarity matrix between all movie vectors.
5. **Recommendation**: For a input movie, find its index, sort similar movies by score (top 5, excluding itself).

This is a classic **TF-IDF + Cosine Similarity** pipeline for text-based recommendation.

## Requirements
- Python 3.6+
- Libraries (install via pip if needed):
  ```
  numpy
  pandas
  matplotlib
  scikit-learn
  ```
- Jupyter Notebook (for running the provided `.ipynb` file).

## Installation
1. Clone or download the project folder.
2. Ensure the dataset files (`tmdb_5000_movies.csv` and `tmdb_5000_credits.csv`) are in the `dataset/` subdirectory.
3. Install dependencies:
   ```
   pip install numpy pandas matplotlib scikit-learn
   ```
4. Launch Jupyter:
   ```
   jupyter notebook
   ```

## Usage
1. Open `movie_recommendation.ipynb` in Jupyter Notebook.
2. Run all cells sequentially (or use "Run All").
3. In the final cell, call the `recommend()` function with a movie title, e.g.:
   ```python
   recommend("Avatar")
   ```
4. Output: Prints top 5 similar movies (e.g., "Aliens to Mars", "Moonraker").

### Example Output
For input: **Avatar**  
Recommendations:  
- Aliens to Mars  
- Moonraker  
- Silent Running  
- Spacballs  
- (One more based on similarity scores)

## Limitations
- Relies solely on textual tags; does not incorporate user ratings or collaborative filtering.
- Fixed max features (5000) may limit vocabulary for larger datasets.
- Case-sensitive title matching; ensure exact spelling.

## Future Improvements
- Integrate collaborative filtering (e.g., using Surprise library).
- Add user profiles for personalized recommendations.
- Visualize similarity matrix with heatmaps.
- Deploy as a web app (e.g., via Streamlit).

## Author
- **Name**: Disha Choudhary
- **Email**: 1530disha@gmail.com
- **Project for**: Acmegrade Artificial Intelligence Course


