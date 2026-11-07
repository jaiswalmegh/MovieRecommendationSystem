# 🎬 CineMatch | AI-Powered Movie Recommendation System

### Discover Your Next Obsession.

CineMatch is an AI-powered movie recommendation platform designed to make movie discovery smarter, faster, and more personalized.

Powered by a **Hybrid Recommendation System**, CineMatch combines Content-Based Filtering and Collaborative Filtering to help users discover movies based on movie characteristics and user preferences.

The platform features a cinematic **After Dark** interface with an obsidian-black background, acid-lime accents, interactive movie cards, animated elements, and a modern Streamlit experience.

---

## ✨ Features

### 🎯 Hybrid Movie Recommendations
- Combines Content-Based and Collaborative Filtering.
- Generates personalized movie recommendations.
- Uses a ranking mechanism to organize recommendation results.
- Supports Top-N recommendation generation.

### 🔎 Explore Movies
- Browse the movie collection.
- Search movies by title.
- Filter movies by genre.
- Filter by minimum rating and number of votes.
- Sort by popularity, rating, title, and release year.
- Paginated movie discovery.
- Interactive movie selection and details.

### 🧠 Model Insights
- Dataset statistics and rating distribution.
- Hybrid model weight information.
- Content-Based model insights.
- Collaborative Filtering model insights.
- Recommendation architecture visualization.
- Model evaluation results when available.

### 🎨 CineMatch After Dark UI
- Obsidian-black visual theme.
- Acid-lime and electric-cyan accents.
- Animated hero section.
- Interactive movie cards.
- Hover effects and transitions.
- Responsive layouts.
- Custom typography and cinematic styling.

---

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Frontend | Streamlit, HTML, CSS |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Recommendation System | Content-Based Filtering, Collaborative Filtering |
| Feature Engineering | TF-IDF, Cosine Similarity |
| Visualization | Matplotlib, Seaborn, Streamlit Charts |
| Model Persistence | Joblib |
| Dataset | MovieLens 100K |

---

## 🧠 Recommendation System Architecture

CineMatch uses a hybrid approach to combine two recommendation techniques.

### 1. Content-Based Filtering

Recommends movies based on their characteristics and similarity.

- Movie feature extraction.
- TF-IDF feature representation.
- Cosine similarity.
- Similar movie discovery.

### 2. Collaborative Filtering

Uses user-movie rating interactions to identify preference patterns.

- User-item interaction matrix.
- User preference analysis.
- Similarity-based recommendation.
- Rating-based signals.

### 3. Hybrid Recommendation Engine

Combines the outputs of both recommendation approaches using configurable weights.

Conceptually:

\[
HybridScore = (W_c \times ContentScore)
+ (W_{cf} \times CollaborativeScore)
\]

Where:

- \(W_c\) = Content-Based weight
- \(W_{cf}\) = Collaborative Filtering weight

The final recommendation list is processed through the ranking system to generate Top-N results.

---

## 📂 Project Structure

```text
CineMatch/
│
├── app/
│   ├── dashboard.py
│   │
│   ├── pages/
│   │   ├── home.py
│   │   ├── explore_movies.py
│   │   ├── recommendations.py
│   │   └── model_insights.py
│   │
│   └── components/
│       ├── movie_card.py
│       ├── recommendation_card.py
│       └── charts.py
│
├── assets/
│   ├── styles/
│   │   └── theme.css
│   └── images/
│
├── config/
│   ├── __init__.py
│   └── settings.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
│
├── src/
│   ├── data/
│   │   ├── data_loader.py
│   │   ├── data_cleaner.py
│   │   └── feature_engineering.py
│   │
│   ├── models/
│   │   ├── content_based.py
│   │   ├── collaborative.py
│   │   └── hybrid_model.py
│   │
│   ├── recommendation/
│   │   ├── recommendation_engine.py
│   │   ├── similarity_engine.py
│   │   └── ranking.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── model_evaluator.py
│   │
│   └── utils/
│       ├── helpers.py
│       └── logger.py
│
├── notebooks/
│
├── tests/
│
├── artifacts/
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.10 or newer recommended.
- pip package manager.
- Git.

### 1. Clone the Repository

```bash
git clone https://github.com/jaiswalmegh/MovieRecommendationSystem
```

```bash
cd CineMatch
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run CineMatch

```bash
python main.py
```

The project launcher handles model initialization when required and starts the Streamlit application.

Open the local application at:

```text
http://localhost:8501
```

**Note:** The launcher expects the configured MovieLens dataset and project dependencies to be available. On first execution, the model initialization pipeline may take some time.

---

## 📊 Dataset

CineMatch is built around the MovieLens 100K dataset.

The dataset contains movie information and user-rating interactions used for training and recommendation generation.

Dataset source:

[MovieLens 100K Dataset](https://grouplens.org/datasets/movielens/100k/)

---

## 📈 Model Evaluation

CineMatch includes an evaluation module with recommendation metrics such as:

- Precision@K
- Recall@K
- F1@K
- Hit Rate@K
- NDCG@K
- RMSE
- MAE

Evaluation results can be saved under:

```text
artifacts/model_evaluation_results.csv
```

---

## 🖥️ Application Pages

| Page | Description |
|---|---|
| Home | Platform overview, statistics and trending movies |
| Explore Movies | Movie search, filtering, sorting and discovery |
| Recommendations | Personalized movie recommendations |
| Model Insights | Dataset, model architecture and evaluation insights |

---

## 🔮 Future Enhancements

- Real movie posters and movie artwork integration.
- Detailed movie information pages.
- Enhanced recommendation explanations.
- Advanced recommendation ranking.
- Improved recommendation evaluation.
- User authentication and personalized profiles.
- Watchlist and favourites.
- Additional recommendation algorithms.
- Deployment to a cloud platform.

---

## 🎯 Project Objective

The primary objective of CineMatch is to demonstrate how Machine Learning and recommendation algorithms can improve personalized content discovery.

The project brings together data preprocessing, feature engineering, machine learning, recommendation algorithms, evaluation metrics, and frontend development in one integrated application.

---

## 👨‍💻 Developer

**Megh Jaiswal**

B.Tech | Computer Science & Engineering (AI)

---

### Built with Python. Powered by Machine Learning. Designed for Movie Lovers. 🎬

**CineMatch | Your Next Obsession.**
