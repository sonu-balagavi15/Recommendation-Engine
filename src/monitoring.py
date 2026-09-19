import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Load dataset
# ==========================================

df = pd.read_csv("data/interactions.csv")

print("=" * 60)
print("RECOMMENDATION SYSTEM MONITORING")
print("=" * 60)

# ==========================================
# 2. Basic monitoring metrics
# ==========================================

print("\n--- Basic System Metrics ---")

print(f"Total interactions: {len(df)}")
print(f"Unique users: {df['user_id'].nunique()}")
print(f"Unique items: {df['item_id'].nunique()}")
print(f"Average rating: {df['rating'].mean():.3f}")

# ==========================================
# 3. Rating distribution
# ==========================================

rating_distribution = (
    df["rating"]
    .value_counts(normalize=True)
    .sort_index()
)

print("\n--- Rating Distribution ---")
print((rating_distribution * 100).round(2))

# ==========================================
# 4. User activity monitoring
# ==========================================

user_activity = df.groupby("user_id").size()

print("\n--- User Activity ---")

print(
    f"Average interactions/user: "
    f"{user_activity.mean():.2f}"
)

print(
    f"Median interactions/user: "
    f"{user_activity.median():.2f}"
)

print(
    f"Maximum interactions/user: "
    f"{user_activity.max()}"
)

# ==========================================
# 5. Item activity monitoring
# ==========================================

item_activity = df.groupby("item_id").size()

print("\n--- Item Activity ---")

print(
    f"Average interactions/item: "
    f"{item_activity.mean():.2f}"
)

print(
    f"Median interactions/item: "
    f"{item_activity.median():.2f}"
)

print(
    f"Maximum interactions/item: "
    f"{item_activity.max()}"
)

# ==========================================
# 6. Detect inactive users
# ==========================================

inactive_users = (
    user_activity < 3
).sum()

inactive_percentage = (
    inactive_users
    / len(user_activity)
    * 100
)

print("\n--- Inactive Users ---")

print(
    f"Users with fewer than 3 interactions: "
    f"{inactive_users}"
)

print(
    f"Inactive user percentage: "
    f"{inactive_percentage:.2f}%"
)

# ==========================================
# 7. Detect low-activity items
# ==========================================

low_activity_items = (
    item_activity < 5
).sum()

print("\n--- Low Activity Items ---")

print(
    f"Items with fewer than 5 interactions: "
    f"{low_activity_items}"
)

# ==========================================
# 8. Rating drift simulation
# ==========================================

# Split data into two time periods

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

median_time = df["timestamp"].median()

baseline = df[
    df["timestamp"] <= median_time
]

recent = df[
    df["timestamp"] > median_time
]

baseline_rating = baseline["rating"].mean()
recent_rating = recent["rating"].mean()

rating_drift = (
    recent_rating - baseline_rating
)

print("\n--- Rating Drift ---")

print(
    f"Baseline average rating: "
    f"{baseline_rating:.3f}"
)

print(
    f"Recent average rating: "
    f"{recent_rating:.3f}"
)

print(
    f"Rating drift: "
    f"{rating_drift:.3f}"
)

# ==========================================
# 9. Drift threshold
# ==========================================

DRIFT_THRESHOLD = 0.20

if abs(rating_drift) > DRIFT_THRESHOLD:

    print(
        "\n⚠️ ALERT: Significant rating drift detected!"
    )

else:

    print(
        "\n✓ No significant rating drift detected."
    )

# ==========================================
# 10. Visualization
# ==========================================

plt.figure(figsize=(8, 5))

plt.bar(
    ["Baseline", "Recent"],
    [baseline_rating, recent_rating]
)

plt.title("Average Rating Drift")
plt.ylabel("Average Rating")

plt.tight_layout()

plt.savefig(
    "outputs/rating_drift.png"
)

plt.show()

print("\nMonitoring completed successfully!")