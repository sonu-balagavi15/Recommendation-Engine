# Recommendation Engine with Monitoring

A machine learning-based recommendation system that provides personalized item recommendations for existing users and popularity-based recommendations for new users. The project also includes model evaluation, monitoring, data drift detection, and a Flask REST API for serving recommendations.

---

## 🚀 Live Demo

**Live API:**  
https://recommendation-engine-h5gc.onrender.com

**GitHub Repository:**  
https://github.com/sonu-balagavi15/Recommendation-Engine

---

## 📌 Project Overview

Recommendation systems are widely used by platforms such as e-commerce websites, streaming services, social media platforms, and online learning platforms to suggest relevant content or products to users.

This project implements a complete recommendation pipeline using collaborative filtering and dimensionality reduction with **Truncated SVD**.

The system supports two major recommendation scenarios:

1. **Personalized Recommendations**
   - Used when a user already has interaction history.
   - The system learns user-item relationships.
   - Previously interacted items are excluded.
   - New items are ranked based on predicted scores.

2. **Cold-Start Recommendations**
   - Used when a new user has no interaction history.
   - The system uses popularity and weighted average ratings.
   - This allows the recommendation service to provide useful results even for new users.

The project also includes:

- Data exploration
- Machine learning model training
- Model evaluation
- Recommendation generation
- Cold-start handling
- Monitoring
- Data drift detection
- REST API
- GitHub version control
- Cloud deployment using Render

---

# 🎯 Objectives

The main objectives of this project are:

- Build a machine learning recommendation engine.
- Analyze user-item interaction data.
- Implement collaborative filtering.
- Use Truncated SVD for dimensionality reduction.
- Generate personalized recommendations.
- Handle new users using a cold-start strategy.
- Evaluate recommendation performance.
- Monitor recommendation data.
- Detect changes in data distribution.
- Expose the recommendation system through a REST API.
- Deploy the application to the cloud.

---

# 📊 Dataset

The project uses a generated interaction dataset for demonstrating the recommendation pipeline.

### Dataset Statistics

| Feature | Value |
|---|---:|
| Total Interactions | 29,101 |
| Unique Users | 1,000 |
| Unique Items | 500 |
| Rating Scale | 1–5 |
| Average Rating | 3.61 |

### Dataset Columns

| Column | Description |
|---|---|
| `user_id` | Unique identifier of the user |
| `item_id` | Unique identifier of the item |
| `rating` | User rating for an item |
| `timestamp` | Time of the interaction |

Dataset file:

```text
data/interactions.csv
