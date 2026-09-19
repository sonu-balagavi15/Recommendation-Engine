import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD

# ==============================
# 1. Load dataset
# ==============================

df = pd.read_csv("data/interactions.csv")

print("Dataset loaded:", df.shape)

# ==============================
# 2. Encode users and items
# ==============================

user_encoder = LabelEncoder()
item_encoder = LabelEncoder()

df["user_encoded"] = user_encoder.fit_transform(df["user_id"])
df["item_encoded"] = item_encoder.fit_transform(df["item_id"])

num_users = df["user_encoded"].nunique()
num_items = df["item_encoded"].nunique()

print("Users:", num_users)
print("Items:", num_items)

# ==============================
# 3. Train-test split
# ==============================

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

print("Training interactions:", len(train_df))
print("Testing interactions:", len(test_df))

# ==============================
# 4. Create User-Item Matrix
# ==============================

user_item_matrix = train_df.pivot_table(
    index="user_encoded",
    columns="item_encoded",
    values="rating",
    fill_value=0
)

print("User-Item Matrix:", user_item_matrix.shape)

# ==============================
# 5. Matrix Factorization
# ==============================

n_components = 30

svd = TruncatedSVD(
    n_components=n_components,
    random_state=42
)

user_factors = svd.fit_transform(user_item_matrix)

item_factors = svd.components_

print("Latent dimensions:", n_components)

# ==============================
# 6. Reconstruct predicted ratings
# ==============================

predicted_matrix = np.dot(
    user_factors,
    item_factors
)

# ==============================
# 7. Save model
# ==============================

joblib.dump(
    svd,
    "models/recommendation_model.joblib"
)

joblib.dump(
    user_encoder,
    "models/user_encoder.joblib"
)

joblib.dump(
    item_encoder,
    "models/item_encoder.joblib"
)

joblib.dump(
    user_item_matrix,
    "models/user_item_matrix.joblib"
)

joblib.dump(
    predicted_matrix,
    "models/predicted_matrix.joblib"
)

print("\nModel training completed successfully!")

print("\nFiles saved:")
print("models/recommendation_model.joblib")
print("models/user_encoder.joblib")
print("models/item_encoder.joblib")
print("models/user_item_matrix.joblib")
print("models/predicted_matrix.joblib")