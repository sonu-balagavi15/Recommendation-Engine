import pandas as pd
import numpy as np
import joblib


# ==========================================
# Load dataset and trained files
# ==========================================

df = pd.read_csv("data/interactions.csv")

predicted_matrix = joblib.load(
    "models/predicted_matrix.joblib"
)

user_encoder = joblib.load(
    "models/user_encoder.joblib"
)

item_encoder = joblib.load(
    "models/item_encoder.joblib"
)


# ==========================================
# Cold-start recommendation
# ==========================================

def cold_start_recommendations(top_n=10):

    # Calculate statistics for each item
    item_stats = (
        df.groupby("item_id")
        .agg(
            average_rating=("rating", "mean"),
            rating_count=("rating", "count")
        )
        .reset_index()
    )

    # Minimum number of ratings
    min_ratings = 10

    popular_items = item_stats[
        item_stats["rating_count"] >= min_ratings
    ].copy()

    # Calculate a weighted score
    global_average = df["rating"].mean()

    popular_items["weighted_score"] = (
        (
            popular_items["rating_count"]
            / (
                popular_items["rating_count"]
                + min_ratings
            )
        )
        * popular_items["average_rating"]
        +
        (
            min_ratings
            / (
                popular_items["rating_count"]
                + min_ratings
            )
        )
        * global_average
    )

    # Sort by weighted score
    popular_items = popular_items.sort_values(
        "weighted_score",
        ascending=False
    )

    return popular_items.head(top_n)


# ==========================================
# Test cold-start system
# ==========================================

new_user_id = 99999

print("=" * 60)
print("COLD-START RECOMMENDATION")
print("=" * 60)

print(f"\nNew user: {new_user_id}")

if new_user_id not in user_encoder.classes_:

    print(
        "\nUser has no interaction history."
    )

    print(
        "Using popular high-rated items as fallback..."
    )

    recommendations = cold_start_recommendations(10)

    print("\nRecommended items:")

    print(
        recommendations[
            [
                "item_id",
                "average_rating",
                "rating_count",
                "weighted_score"
            ]
        ].to_string(index=False)
    )

else:

    print(
        "\nUser already exists in the system."
    )