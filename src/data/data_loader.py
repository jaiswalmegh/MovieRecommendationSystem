
"""
CineMatch - Data Loader
Downloads and loads the MovieLens 100K dataset.
"""

import zipfile
import urllib.request
from pathlib import Path

import pandas as pd

from config.settings import (
    DATASET_URL,
    RAW_DATA_DIR,
    MOVIES_FILE,
    RATINGS_FILE,
    USERS_FILE,
    GENRES_FILE,
)


class MovieLensDataLoader:
    """Handles downloading and loading the MovieLens 100K dataset."""

    MOVIE_COLUMNS = [
        "movie_id",
        "title",
        "release_date",
        "video_release_date",
        "imdb_url",
        "unknown",
        "action",
        "adventure",
        "animation",
        "childrens",
        "comedy",
        "crime",
        "documentary",
        "drama",
        "fantasy",
        "film_noir",
        "horror",
        "musical",
        "mystery",
        "romance",
        "sci_fi",
        "thriller",
        "war",
        "western",
    ]

    RATING_COLUMNS = [
        "user_id",
        "movie_id",
        "rating",
        "timestamp",
    ]

    USER_COLUMNS = [
        "user_id",
        "age",
        "gender",
        "occupation",
        "zip_code",
    ]

    GENRE_COLUMNS = ["genre", "genre_id"]

    def __init__(self):
        self.raw_dir = Path(RAW_DATA_DIR)
        self.movies = None
        self.ratings = None
        self.users = None
        self.genres = None

    def dataset_exists(self):
        """Check whether all required dataset files exist."""

        required_files = [
            MOVIES_FILE,
            RATINGS_FILE,
            USERS_FILE,
            GENRES_FILE,
        ]

        return all(Path(file).exists() for file in required_files)

    def download_dataset(self):
        """Download and extract MovieLens 100K."""

        self.raw_dir.mkdir(parents=True, exist_ok=True)

        zip_path = self.raw_dir / "ml-100k.zip"

        print("Downloading MovieLens 100K dataset...")

        try:
            urllib.request.urlretrieve(DATASET_URL, zip_path)

            print("Download completed.")
            print("Extracting dataset...")

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(self.raw_dir)

            zip_path.unlink(missing_ok=True)

            print("Dataset extracted successfully.")

        except Exception as error:
            raise RuntimeError(
                f"Dataset download failed: {error}"
            ) from error

        if not self.dataset_exists():
            raise FileNotFoundError(
                "Dataset extraction completed, but required files "
                "were not found. Please check the dataset archive."
            )

    def ensure_dataset_available(self):
        """Download the dataset if it is not already available."""

        if not self.dataset_exists():
            print("Dataset not found locally.")
            self.download_dataset()
        else:
            print("MovieLens dataset already exists.")

    def load_movies(self):
        """Load movie information."""

        if not Path(MOVIES_FILE).exists():
            raise FileNotFoundError(
                f"Movies file not found: {MOVIES_FILE}"
            )

        self.movies = pd.read_csv(
            MOVIES_FILE,
            sep="|",
            encoding="latin-1",
            names=self.MOVIE_COLUMNS,
            engine="python",
            usecols=range(24),
        )

        self.movies["movie_id"] = self.movies["movie_id"].astype(int)
        self.movies["title"] = self.movies["title"].fillna("Unknown")

        return self.movies

    def load_ratings(self):
        """Load user ratings."""

        if not Path(RATINGS_FILE).exists():
            raise FileNotFoundError(
                f"Ratings file not found: {RATINGS_FILE}"
            )

        self.ratings = pd.read_csv(
            RATINGS_FILE,
            sep="\t",
            names=self.RATING_COLUMNS,
            encoding="latin-1",
        )

        self.ratings["user_id"] = self.ratings["user_id"].astype(int)
        self.ratings["movie_id"] = self.ratings["movie_id"].astype(int)
        self.ratings["rating"] = self.ratings["rating"].astype(float)

        return self.ratings

    def load_users(self):
        """Load user demographic information."""

        if not Path(USERS_FILE).exists():
            raise FileNotFoundError(
                f"Users file not found: {USERS_FILE}"
            )

        self.users = pd.read_csv(
            USERS_FILE,
            sep="|",
            names=self.USER_COLUMNS,
            encoding="latin-1",
        )

        self.users["user_id"] = self.users["user_id"].astype(int)
        self.users["age"] = self.users["age"].astype(int)

        return self.users

    def load_genres(self):
        """Load available movie genres."""

        if not Path(GENRES_FILE).exists():
            raise FileNotFoundError(
                f"Genres file not found: {GENRES_FILE}"
            )

        self.genres = pd.read_csv(
            GENRES_FILE,
            sep="|",
            names=self.GENRE_COLUMNS,
            encoding="latin-1",
        )

        return self.genres

    def load_all_data(self, download_if_missing=True):
        """
        Load all dataset files.

        Returns:
            dict: Movies, ratings, users and genres DataFrames.
        """

        if download_if_missing:
            self.ensure_dataset_available()
        elif not self.dataset_exists():
            raise FileNotFoundError(
                "MovieLens dataset is missing. "
                "Run download_dataset() first."
            )

        print("\nLoading MovieLens data...")

        movies = self.load_movies()
        ratings = self.load_ratings()
        users = self.load_users()
        genres = self.load_genres()

        print("\nDataset loaded successfully!")
        print(f"Movies  : {len(movies)}")
        print(f"Ratings : {len(ratings)}")
        print(f"Users   : {len(users)}")
        print(f"Genres  : {len(genres)}")

        return {
            "movies": movies,
            "ratings": ratings,
            "users": users,
            "genres": genres,
        }


if __name__ == "__main__":
    loader = MovieLensDataLoader()

    data = loader.load_all_data()

    print("\nMovies Preview:")
    print(data["movies"][["movie_id", "title"]].head())

    print("\nRatings Preview:")
    print(data["ratings"].head())
