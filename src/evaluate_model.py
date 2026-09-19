import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error


# ==========================================
# 1. Load data and model
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

user_item_matrix = joblib.load(
    "models/user_item_matrix.joblib"
)


# ==========================================
# 2. Create train/test split
# ==========================================

np.random.seed(42)

test_mask = np.random.rand(len(df)) < 0.2

test_df = df[test_mask].copy()


# ==========================================
# 3. RMSE evaluation
# ==========================================

actual_ratings = []
predicted_ratings = []

for _, row in test_df.iterrows():

    user_id = row["user_id"]
    item_id = row["item_id"]

    if user_id not in user_encoder.classes_:
        continue

    if item_id not in item_encoder.classes_:
        continue

    user_index = user_encoder.transform(
        [user_id]
    )[0]

    item_index = item_encoder.transform(
        [item_id]
    )[0]

    actual_ratings.append(row["rating"])

    predicted_ratings.append(
        predicted_matrix[user_index][item_index]
    )


rmse = np.sqrt(
    mean_squared_error(
        actual_ratings,
        predicted_ratings
    )
)

print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(f"\nRMSE: {rmse:.4f}")


# ==========================================
# 4. Precision@10 and Recall@10
# ==========================================

K = 10

precisions = []
recalls = []


# Group test interactions by user
test_users = test_df["user_id"].unique()

for user_id in test_users:

    if user_id not in user_encoder.classes_:
        continue

    user_index = user_encoder.transform(
        [user_id]
    )[0]

    # Items actually liked by the user
    relevant_items = set(
        test_df[
            (test_df["user_id"] == user_id)
            & (test_df["rating"] >= 4)
        ]["item_id"]
    )

    if len(relevant_items) == 0:
        continue

    # Predicted ratings
    predictions = predicted_matrix[user_index]

    # Items already present in training data
    interacted_items = set(
        user_item_matrix.loc[user_index]
        .loc[lambda x: x > 0]
        .index
    )

    # Rank items
    ranked_items = np.argsort(
        predictions
    )[::-1]

    recommendations = []

    for item_index in ranked_items:

        if item_index in interacted_items:
            continue

        original_item = item_encoder.inverse_transform(
            [item_index]
        )[0]

        recommendations.append(
            original_item
        )

        if len(recommendations) == K:
            break

    recommended_set = set(recommendations)

    # Calculate hits
    hits = len(
        recommended_set.intersection(
            relevant_items
        )
    )

    precision = hits / K

    recall = hits / len(relevant_items)

    precisions.append(precision)
    recalls.append(recall)


# ==========================================
# 5. Final metrics
# ==========================================

precision_at_10 = np.mean(precisions)

recall_at_10 = np.mean(recalls)

print(
    f"Precision@10: {precision_at_10:.4f}"
)

print(
    f"Recall@10: {recall_at_10:.4f}"
)

print(
    f"\nUsers evaluated: {len(precisions)}"
)

print("\nEvaluation completed successfully!")