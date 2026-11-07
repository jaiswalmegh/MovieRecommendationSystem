
"""
CineMatch - Helper Utilities

Contains reusable functions for data formatting,
movie processing, rating conversion and validation.
"""

import re
import numpy as np
import pandas as pd


# ==========================================
# MOVIE UTILITIES
# ==========================================

def clean_movie_title(title):
    """Clean and normalize movie titles."""

    if pd.isna(title):
        return "Unknown"

    title = str(title).strip()

    title = re.sub(r"\s+", " ", title)

    return title


def extract_release_year(title):
    """Extract release year from a movie title."""

    if pd.isna(title):
        return None

    match = re.search(r"\((\d{4})\)\s*$", str(title))

    if match:
        return int(match.group(1))

    return None


def format_movie_title(title):
    """Return a display-friendly movie title."""

    title = clean_movie_title(title)

    return title.title()


# ==========================================
# RATING UTILITIES
# ==========================================

def normalize_rating(rating, minimum=1, maximum=5):
    """Normalize a rating to the range 0-1."""

    try:
        rating = float(rating)
    except (TypeError, ValueError):
        raise ValueError("Rating must be numeric.")

    if not np.isfinite(rating):
        raise ValueError("Rating must be finite.")

    if not minimum <= rating <= maximum:
        raise ValueError(
            f"Rating must be between {minimum} and {maximum}."
        )

    if maximum <= minimum:
        raise ValueError("Maximum must be greater than minimum.")

    return (rating - minimum) / (maximum - minimum)


def denormalize_rating(
    normalized_rating,
    minimum=1,
    maximum=5,
):
    """Convert a normalized rating back to its original scale."""

    try:
        normalized_rating = float(normalized_rating)
    except (TypeError, ValueError):
        raise ValueError(
            "Normalized rating must be numeric."
        )

    if not np.isfinite(normalized_rating):
        raise ValueError(
            "Normalized rating must be finite."
        )

    if not 0 <= normalized_rating <= 1:
        raise ValueError(
            "Normalized rating must be between 0 and 1."
        )

    return minimum + normalized_rating * (
        maximum - minimum
    )


def get_rating_label(rating):
    """Convert numerical ratings into descriptive labels."""

    if pd.isna(rating):
        return "Not Rated"

    rating = float(rating)

    if rating >= 4.5:
        return "Excellent"

    if rating >= 4.0:
        return "Very Good"

    if rating >= 3.0:
        return "Good"

    if rating >= 2.0:
        return "Average"

    return "Poor"


# ==========================================
# DATAFRAME UTILITIES
# ==========================================

def validate_dataframe(dataframe, required_columns):
    """Validate DataFrame and required columns."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "Input must be a pandas DataFrame."
        )

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    return True


def remove_duplicate_records(
    dataframe,
    subset=None,
):
    """Remove duplicate records from a DataFrame."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "Input must be a pandas DataFrame."
        )

    return dataframe.drop_duplicates(
        subset=subset
    ).reset_index(drop=True)


def get_dataframe_summary(dataframe):
    """Generate a compact DataFrame summary."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "Input must be a pandas DataFrame."
        )

    return {
        "rows": len(dataframe),
        "columns": len(dataframe.columns),
        "missing_values": int(
            dataframe.isna().sum().sum()
        ),
        "duplicate_rows": int(
            dataframe.duplicated().sum()
        ),
        "memory_usage_mb": round(
            dataframe.memory_usage(deep=True).sum()
            / (1024 ** 2),
            3,
        ),
    }


# ==========================================
# GENRE UTILITIES
# ==========================================

def extract_movie_genres(movie, genre_columns):
    """Extract active genres from a movie record."""

    return [
        genre
        for genre in genre_columns
        if genre in movie.index
        and movie[genre] == 1
    ]


def format_genre_list(genres):
    """Format genre names for display."""

    if not genres:
        return "Unknown"

    return ", ".join(
        str(genre).replace("_", " ").title()
        for genre in genres
    )


# ==========================================
# RECOMMENDATION UTILITIES
# ==========================================

def remove_watched_movies(
    recommendations,
    watched_movie_ids,
):
    """Remove already watched movies from recommendations."""

    if "movie_id" not in recommendations.columns:
        raise ValueError(
            "Recommendations must contain movie_id."
        )

    watched_ids = set(watched_movie_ids)

    return (
        recommendations[
            ~recommendations["movie_id"].isin(watched_ids)
        ]
        .reset_index(drop=True)
    )


def format_recommendation_scores(
    recommendations,
    score_column="hybrid_score",
):
    """Format recommendation scores for presentation."""

    if score_column not in recommendations.columns:
        raise ValueError(
            f"Missing score column: {score_column}"
        )

    result = recommendations.copy()

    result[score_column] = pd.to_numeric(
        result[score_column],
        errors="coerce",
    )

    result[f"{score_column}_percentage"] = (
        result[score_column] * 100
    ).round(2)

    return result


# ==========================================
# GENERAL UTILITIES
# ==========================================

def safe_divide(
    numerator,
    denominator,
    default=0.0,
):
    """Perform division safely."""

    if denominator == 0:
        return default

    return numerator / denominator


def format_large_number(number):
    """Format large numbers for dashboard display."""

    number = float(number)

    if number >= 1_000_000:
        return f"{number / 1_000_000:.1f}M"

    if number >= 1_000:
        return f"{number / 1_000:.1f}K"

    return str(int(number))


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":

    print("=" * 50)
    print("       CINEMATCH HELPER UTILITIES")
    print("=" * 50)

    print("\nMovie Title Utilities:")

    title = "Toy Story (1995)"

    print("Original:", title)
    print("Cleaned:", clean_movie_title(title))
    print("Year:", extract_release_year(title))
    print("Formatted:", format_movie_title(title))

    print("\nRating Utilities:")

    rating = 4.5

    normalized = normalize_rating(rating)

    print("Original Rating:", rating)
    print("Normalized:", normalized)
    print(
        "Denormalized:",
        denormalize_rating(normalized),
    )

    print("Rating Label:", get_rating_label(rating))

    print("\nGeneral Utilities:")

    print("Safe Division:", safe_divide(10, 2))
    print("Formatted Number:", format_large_number(12500))

    print("\nHelper utilities executed successfully!")
