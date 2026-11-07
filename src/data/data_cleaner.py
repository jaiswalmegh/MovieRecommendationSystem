
"""
CineMatch - Data Cleaner
Cleans and prepares MovieLens 100K data.
"""

from pathlib import Path
import pandas as pd

from config.settings import PROCESSED_DATA_DIR


class MovieLensDataCleaner:

    RATING_MIN = 1
    RATING_MAX = 5

    GENRE_COLUMNS = [
        "unknown", "action", "adventure", "animation",
        "childrens", "comedy", "crime", "documentary",
        "drama", "fantasy", "film_noir", "horror",
        "musical", "mystery", "romance", "sci_fi",
        "thriller", "war", "western"
    ]

    def __init__(self):
        self.output_dir = Path(PROCESSED_DATA_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def clean_movies(self, movies):

        print("Cleaning movie data...")

        movies = movies.copy()

        movies = movies.drop_duplicates(
            subset=["movie_id"]
        )

        movies["title"] = (
            movies["title"]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
        )

        movies["release_date"] = pd.to_datetime(
            movies["release_date"],
            errors="coerce"
        )

        for column in self.GENRE_COLUMNS:
            if column in movies.columns:
                movies[column] = pd.to_numeric(
                    movies[column],
                    errors="coerce"
                ).fillna(0).astype(int)

        movies = movies[
            movies["movie_id"].notna()
            & (movies["movie_id"] > 0)
        ]

        return movies.reset_index(drop=True)

    def clean_ratings(self, ratings):

        print("Cleaning ratings data...")

        ratings = ratings.copy()

        ratings["rating"] = pd.to_numeric(
            ratings["rating"],
            errors="coerce"
        )

        ratings = ratings.dropna(
            subset=["user_id", "movie_id", "rating"]
        )

        ratings = ratings[
            (ratings["rating"] >= self.RATING_MIN)
            & (ratings["rating"] <= self.RATING_MAX)
            & (ratings["user_id"] > 0)
            & (ratings["movie_id"] > 0)
        ]

        ratings = ratings.drop_duplicates(
            subset=["user_id", "movie_id"],
            keep="last"
        )

        ratings["user_id"] = ratings["user_id"].astype(int)
        ratings["movie_id"] = ratings["movie_id"].astype(int)

        return ratings.reset_index(drop=True)

    def clean_users(self, users):

        print("Cleaning user data...")

        users = users.copy()

        users["age"] = pd.to_numeric(
            users["age"],
            errors="coerce"
        )

        users = users.dropna(
            subset=["user_id", "age"]
        )

        users = users[
            (users["user_id"] > 0)
            & (users["age"] >= 1)
            & (users["age"] <= 120)
        ]

        users = users.drop_duplicates(
            subset=["user_id"]
        )

        users["user_id"] = users["user_id"].astype(int)
        users["age"] = users["age"].astype(int)

        users["gender"] = (
            users["gender"]
            .fillna("Unknown")
            .astype(str)
            .str.upper()
            .str.strip()
        )

        users["occupation"] = (
            users["occupation"]
            .fillna("unknown")
            .astype(str)
            .str.lower()
            .str.strip()
        )

        users["zip_code"] = (
            users["zip_code"]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
        )

        return users.reset_index(drop=True)

    def clean_genres(self, genres):

        print("Cleaning genre data...")

        genres = genres.copy()

        genres["genre_id"] = pd.to_numeric(
            genres["genre_id"],
            errors="coerce"
        )

        genres = genres.dropna(
            subset=["genre_id"]
        )

        genres["genre_id"] = genres["genre_id"].astype(int)

        genres["genre"] = (
            genres["genre"]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
        )

        genres = genres.drop_duplicates(
            subset=["genre_id"]
        )

        return genres.reset_index(drop=True)

    def validate_data(self, movies, ratings, users):

        print("Validating datasets...")

        movie_ids = set(movies["movie_id"])
        user_ids = set(users["user_id"])

        ratings = ratings[
            ratings["movie_id"].isin(movie_ids)
            & ratings["user_id"].isin(user_ids)
        ]

        if movies.empty:
            raise ValueError("Movie dataset is empty.")

        if ratings.empty:
            raise ValueError("Ratings dataset is empty.")

        if users.empty:
            raise ValueError("User dataset is empty.")

        return ratings.reset_index(drop=True)

    def save_data(self, dataframe, filename):

        file_path = self.output_dir / filename

        dataframe.to_csv(
            file_path,
            index=False,
            encoding="utf-8"
        )

        print(f"Saved: {file_path}")

        return file_path

    def clean_all_data(self, data):

        print("\n" + "=" * 50)
        print("       CINEMATCH DATA CLEANING")
        print("=" * 50)

        movies = self.clean_movies(data["movies"])
        ratings = self.clean_ratings(data["ratings"])
        users = self.clean_users(data["users"])
        genres = self.clean_genres(data["genres"])

        ratings = self.validate_data(
            movies,
            ratings,
            users
        )

        cleaned_data = {
            "movies": movies,
            "ratings": ratings,
            "users": users,
            "genres": genres
        }

        self.save_data(movies, "movies_cleaned.csv")
        self.save_data(ratings, "ratings_cleaned.csv")
        self.save_data(users, "users_cleaned.csv")
        self.save_data(genres, "genres_cleaned.csv")

        print("\nDATA CLEANING COMPLETED!")
        print("\nDataset Summary:")

        for name, dataframe in cleaned_data.items():
            print(f"{name}: {dataframe.shape}")

        return cleaned_data


if __name__ == "__main__":

    from src.data.data_loader import MovieLensDataLoader

    loader = MovieLensDataLoader()
    raw_data = loader.load_all_data()

    cleaner = MovieLensDataCleaner()
    cleaner.clean_all_data(raw_data)
