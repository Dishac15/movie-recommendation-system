# 🎬 Movie Recommendation System (Web App)

A **content-based movie recommendation web application** built using **Python, Machine Learning, and Streamlit**.  
The app recommends movies similar to a selected title by analyzing movie metadata such as **overview, genres, and keywords**.

---

## 🌐 Live Demo
👉 **Web App:** https://<your-streamlit-app-link>

> (Link will be updated after deployment)

---

##✨ Features

Interactive movie recommendations

Fast similarity computation

Clean Streamlit UI

Ready for cloud deployment

---

## 📌 Project Overview
This project uses **Natural Language Processing (NLP)** techniques to recommend movies based on content similarity.  
Textual features are transformed into numerical vectors and compared using cosine similarity.

---

## ⚙️ How It Works
1. Movie metadata is combined (overview, genres, keywords)  
2. Text vectorization using **CountVectorizer**  
3. **Cosine similarity** is used to measure movie similarity  
4. Results are displayed via an interactive Streamlit interface  

---

## 🛠️ Tools & Technologies
- Python  
- Pandas, NumPy  
- Scikit-learn  
- Streamlit  
- TMDb Dataset  

---

##📂 Project Structure
├── app.py
├── requirements.txt
├── dataset/
│   ├── tmdb_5000_movies.csv
│   └── tmdb_5000_credits.csv
├── README.md

---


## 🚀 How to Run Locally

### Clone the repository
```bash
git clone https://github.com/<Dishac15>/movie-recommendation-system.git
cd movie-recommendation-system

###Install dependencies
pip install -r requirements.txt

###Run the app
streamlit run app.py


##🔮 Future Enhancements

User-based recommendations

Movie posters & trailers

Advanced filtering

Custom UI theme


##👩‍💻 Author

Disha
Computer Science Engineering Student