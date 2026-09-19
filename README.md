# Recommendation Engine with Monitoring

A machine learning-based recommendation system that generates personalized recommendations for existing users and popularity-based recommendations for new users using a cold-start strategy.

The project includes model training, evaluation, monitoring, drift detection, and a Flask REST API for serving recommendations.

## Project Overview

The system analyzes user-item interactions and uses collaborative filtering with Truncated SVD to generate personalized recommendations.

For new users without interaction history, the system uses a popularity-based cold-start recommendation strategy.

## Features

- Synthetic recommendation dataset generation
- Exploratory Data Analysis (EDA)
- User-item interaction matrix
- Collaborative filtering using Truncated SVD
- Personalized recommendations
- Cold-start recommendations for new users
- RMSE evaluation
- Precision@10 evaluation
- Recall@10 evaluation
- User and item activity monitoring
- Rating distribution monitoring
- Basic data drift detection
- Flask REST API
- Git and GitHub version control

## System Architecture

```text
                    User Request
                         |
                         v
                  Flask REST API
                         |
                +--------+--------+
                |                 |
          Existing User       New User
                |                 |
                v                 v
       SVD Personalized     Cold-Start
       Recommendations      Strategy
                |                 |
                +--------+--------+
                         |
                         v
                Recommended Items
