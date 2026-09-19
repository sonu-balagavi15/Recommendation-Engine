import numpy as np
import pandas as pd

np.random.seed(42)

# Number of users and items
num_users = 1000
num_items = 500

# Generate user-item interactions
num_interactions = 30000

user_ids = np.random.randint(1, num_users + 1, num_interactions)
item_ids = np.random.randint(1, num_items + 1, num_interactions)

# Generate ratings from 1 to 5
ratings = np.random.choice(
    [1, 2, 3, 4, 5],
    size=num_interactions,
    p=[0.08, 0.12, 0.20, 0.30, 0.30]
)

# Generate timestamps
timestamps = pd.date_range(
    start="2025-01-01",
    periods=num_interactions,
    freq="h"
)

# Create dataframe
df = pd.DataFrame({
    "user_id": user_ids,
    "item_id": item_ids,
    "rating": ratings,
    "timestamp": timestamps
})

# Remove duplicate user-item interactions
df = df.drop_duplicates(
    subset=["user_id", "item_id"],
    keep="last"
)

# Sort by timestamp
df = df.sort_values("timestamp").reset_index(drop=True)

# Save dataset
df.to_csv("data/interactions.csv", index=False)

print("Dataset created successfully!")
print(f"Total interactions: {len(df)}")
print(f"Unique users: {df['user_id'].nunique()}")
print(f"Unique items: {df['item_id'].nunique()}")

print("\nRating distribution:")
print(df["rating"].value_counts().sort_index())

print("\nFirst 5 rows:")
print(df.head())