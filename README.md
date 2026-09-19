# Recommendation Engine with Monitoring

## Project Overview

This project is a machine learning based recommendation engine that generates personalized item recommendations for users.

The system also handles the cold-start problem for new users and includes monitoring features to detect changes in user activity, item activity, rating distribution, and rating drift.

## Features

- Synthetic user-item interaction dataset
- Exploratory Data Analysis
- User-item interaction matrix
- Matrix factorization using Truncated SVD
- Personalized recommendations
- Cold-start recommendations
- Model evaluation
- Recommendation API using Flask
- System monitoring
- Rating drift detection
- User and item activity monitoring

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Flask
- Jupyter Notebook

## Dataset

The dataset contains:

- 1,000 unique users
- 500 unique items
- 29,101 user-item interactions
- Ratings from 1 to 5
- Timestamp information

## Recommendation Approach

The recommendation engine uses matrix factorization with Truncated SVD.

The user-item interaction matrix is transformed into a lower-dimensional representation.

The model then reconstructs the matrix to estimate ratings for items that users have not interacted with.

The highest-ranked unseen items are returned as personalized recommendations.

## Cold-Start Handling

For new users who are not present in the training dataset, the system uses a popularity-based recommendation strategy.

Items are ranked using:

- Average rating
- Number of ratings
- Global average rating

This allows the system to provide recommendations even when there is no previous interaction history for the user.

## Model Evaluation

The system evaluates recommendation quality using:

- RMSE
- Precision@10
- Recall@10

## Monitoring

The monitoring system tracks:

- Total interactions
- Number of users
- Number of items
- Average rating
- Rating distribution
- Average user activity
- Average item activity
- Inactive users
- Low-activity items
- Rating drift

The current dataset contains no significant rating drift.

## API

The recommendation engine provides a Flask REST API.

### Personalized Recommendation

```text
/recommend?user_id=103