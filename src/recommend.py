import pandas as pd
import numpy as np
import joblib

# Load model files
predicted_matrix = joblib.load("models/predicted_matrix.joblib")
user_encoder = joblib.load("models/user_encoder.joblib")
item_encoder = joblib.load("models/item_encoder.joblib")
user_item_matrix = joblib.load("models/user_item_matrix.joblib")

# Original dataset
df = pd.read_csv("data/interactions.csv")


def recommend_items(user_id, top_n=10):

    # Check whether user exists
    if user_id not in user_encoder.classes_:
        print(f"User {user_id} is a cold-start user.")
        print("No previous interaction history found.")
        return

    # Convert original user ID to encoded ID
    user_index = user_encoder.transform([user_id])[0]

    # Predicted ratings for this user
    predictions = predicted_matrix[user_index]

    # Items already interacted with
    interacted_items = set(
        user_item_matrix.loc[user_index]
        .loc[lambda x: x > 0]
        .index
    )

    # Rank items by predicted rating
    ranked_items = np.argsort(predictions)[::-1]

    recommendations = []

    for item_index in ranked_items:

        # Skip items the user already interacted with
        if item_index in interacted_items:
            continue

        original_item_id = item_encoder.inverse_transform(
            [item_index]
        )[0]

        predicted_rating = predictions[item_index]

        recommendations.append({
            "item_id": original_item_id,
            "predicted_rating": round(
                float(predicted_rating), 3
            )
        })

        if len(recommendations) == top_n:
            break

    result = pd.DataFrame(recommendations)

    return result


# Test recommendation
user_id = 103

print("=" * 50)
print(f"TOP 10 RECOMMENDATIONS FOR USER {user_id}")
print("=" * 50)

recommendations = recommend_items(user_id, 10)

print(recommendations)