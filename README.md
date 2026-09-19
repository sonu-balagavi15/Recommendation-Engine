# 🚀 Recommendation Engine with Monitoring

A machine learning-based recommendation engine built using **Python, Collaborative Filtering, Truncated SVD, Scikit-learn, and Flask**.

The system provides personalized recommendations for existing users, handles new users using a cold-start strategy, evaluates recommendation performance, monitors user/item activity and rating drift, and exposes the recommendation system through a live REST API deployed on Render.

---

## 🌐 Live Demo

### Live API

https://recommendation-engine-h5gc.onrender.com

### API Home

https://recommendation-engine-h5gc.onrender.com/

### Personalized Recommendation

https://recommendation-engine-h5gc.onrender.com/recommend?user_id=103

### Cold-Start Recommendation

https://recommendation-engine-h5gc.onrender.com/recommend?user_id=99999

---

## 📂 GitHub Repository

https://github.com/sonu-balagavi15/Recommendation-Engine

---

# 📌 Project Overview

Recommendation systems are widely used by e-commerce platforms, streaming services, social media platforms, and content platforms to recommend relevant products or content to users.

This project implements an end-to-end machine learning recommendation engine using user-item interaction data.

The system learns patterns from historical interactions and generates Top-N recommendations.

It also includes:

- Personalized recommendations
- Collaborative filtering
- Truncated SVD
- Cold-start handling
- Model evaluation
- Data monitoring
- Rating drift detection
- Flask REST API
- Cloud deployment
- GitHub version control

---

# 🎯 Project Objectives

The main objectives of this project are:

1. Generate and analyze user-item interaction data.
2. Build a recommendation system using machine learning.
3. Create a user-item interaction matrix.
4. Apply Truncated SVD for dimensionality reduction.
5. Generate personalized recommendations.
6. Handle new users using a cold-start strategy.
7. Evaluate recommendation performance.
8. Monitor user and item activity.
9. Detect changes in rating behavior.
10. Expose recommendations through a REST API.
11. Deploy the API to the cloud.
12. Provide a complete end-to-end ML project workflow.

---

# 📊 Dataset

A synthetic user-item interaction dataset was generated for this project.

## Dataset Statistics

| Feature | Value |
|---|---:|
| Total Interactions | 29,101 |
| Unique Users | 1,000 |
| Unique Items | 500 |
| Rating Scale | 1–5 |
| Average Rating | 3.61 |

## Dataset Columns

| Column | Description |
|---|---|
| `user_id` | Unique identifier of the user |
| `item_id` | Unique identifier of the item |
| `rating` | User rating for an item |
| `timestamp` | Interaction timestamp |

---

# 🔍 Data Exploration

The dataset was analyzed to understand its structure and quality.

The exploration includes:

- Dataset shape
- Column information
- Missing values
- Unique users
- Unique items
- User activity
- Item activity
- Rating distribution
- Average rating

## Data Quality Results

| Metric | Result |
|---|---:|
| Missing Values | 0 |
| Unique Users | 1,000 |
| Unique Items | 500 |
| Average Interactions/User | 29.10 |
| Minimum Interactions/User | 10 |
| Maximum Interactions/User | 45 |
| Average Interactions/Item | 58.20 |
| Minimum Interactions/Item | 34 |
| Maximum Interactions/Item | 81 |
| Average Rating | 3.614 |

---

# 🧠 Machine Learning Approach

The recommendation system uses **Collaborative Filtering**.

The basic idea is to learn relationships between users and items from historical interactions.

## Machine Learning Workflow

```text
User-Item Interaction Data
          ↓
Data Exploration
          ↓
Train/Test Split
          ↓
User-Item Matrix
          ↓
Truncated SVD
          ↓
Latent Representations
          ↓
Predicted Ratings
          ↓
Recommendation Ranking
          ↓
Top-N Recommendations
