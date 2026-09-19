import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load model files
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

df = pd.read_csv("data/interactions.csv")


def popular_recommendations(top_n=10):

    stats = (
        df.groupby("item_id")
        .agg(
            average_rating=("rating", "mean"),
            rating_count=("rating", "count")
        )
        .reset_index()
    )

    min_ratings = 10
    global_average = df["rating"].mean()

    stats["score"] = (
        (
            stats["rating_count"]
            / (stats["rating_count"] + min_ratings)
        )
        * stats["average_rating"]
        +
        (
            min_ratings
            / (stats["rating_count"] + min_ratings)
        )
        * global_average
    )

    return stats.sort_values(
        "score",
        ascending=False
    ).head(top_n)


def recommend(user_id, top_n=10):

    # Cold-start user
    if user_id not in user_encoder.classes_:

        result = popular_recommendations(top_n)

        return {
            "user_id": user_id,
            "type": "cold_start",
            "recommendations":
                result["item_id"].astype(int).tolist()
        }

    # Existing user
    user_index = user_encoder.transform(
        [user_id]
    )[0]

    predictions = predicted_matrix[user_index]

    interacted = set(
        user_item_matrix.loc[user_index]
        .loc[lambda x: x > 0]
        .index
    )

    ranked = np.argsort(
        predictions
    )[::-1]

    recommendations = []

    for item_index in ranked:

        if item_index in interacted:
            continue

        item_id = item_encoder.inverse_transform(
            [item_index]
        )[0]

        recommendations.append({
            "item_id": int(item_id),
            "predicted_rating": round(
                float(predictions[item_index]),
                3
            )
        })

        if len(recommendations) == top_n:
            break

    return {
        "user_id": user_id,
        "type": "personalized",
        "recommendations": recommendations
    }


@app.route("/")
def home():

    return jsonify({
        "message":
            "Recommendation Engine API is running",
        "endpoints": [
            "/recommend?user_id=103",
            "/recommend?user_id=99999"
        ]
    })


@app.route("/recommend")
def recommendation_api():

    user_id = request.args.get(
        "user_id",
        type=int
    )

    top_n = request.args.get(
        "top_n",
        default=10,
        type=int
    )

    if user_id is None:

        return jsonify({
            "error": "user_id is required"
        }), 400

    if top_n < 1 or top_n > 50:

        return jsonify({
            "error":
                "top_n must be between 1 and 50"
        }), 400

    result = recommend(
        user_id,
        top_n
    )

    return jsonify(result)


if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )