import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/interactions.csv")

print("========== DATASET INFORMATION ==========")

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\n========== USER / ITEM STATISTICS ==========")

print("\nUnique users:")
print(df["user_id"].nunique())

print("\nUnique items:")
print(df["item_id"].nunique())

print("\nInteractions per user:")
print(df.groupby("user_id").size().describe())

print("\nInteractions per item:")
print(df.groupby("item_id").size().describe())

print("\n========== RATING STATISTICS ==========")

print("\nRating counts:")
print(df["rating"].value_counts().sort_index())

print("\nAverage rating:")
print(df["rating"].mean())

print("\n========== TOP ITEMS ==========")

top_items = (
    df.groupby("item_id")
    .agg(
        average_rating=("rating", "mean"),
        rating_count=("rating", "count")
    )
    .sort_values(
        ["rating_count", "average_rating"],
        ascending=False
    )
    .head(10)
)

print(top_items)

# Rating distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="rating")
plt.title("Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Number of Ratings")
plt.tight_layout()
plt.savefig("outputs/rating_distribution.png")
plt.show()

# Interactions per user
user_counts = df.groupby("user_id").size()

plt.figure(figsize=(8, 5))
plt.hist(user_counts, bins=30)
plt.title("Interactions per User")
plt.xlabel("Number of Interactions")
plt.ylabel("Number of Users")
plt.tight_layout()
plt.savefig("outputs/user_interactions.png")
plt.show()

print("\nExploration completed successfully!")