# Recommendation Engine with Monitoring

A machine learning-based recommendation system that generates personalized recommendations for existing users and popularity-based recommendations for new users using a cold-start strategy.

The project includes data generation, exploratory data analysis, collaborative filtering, model evaluation, cold-start handling, monitoring, drift detection, and a Flask REST API for serving recommendations.

---

## 🚀 Project Overview

Recommendation systems are used to suggest relevant items to users based on their previous interactions.

This project demonstrates an end-to-end recommendation pipeline:

```text
User-Item Interaction Data
            ↓
    Data Preprocessing
            ↓
    User-Item Matrix
            ↓
       Truncated SVD
            ↓
   Predicted Item Scores
            ↓
    Recommendation Engine
            ↓
     +---------------+
     |               |
Existing User      New User
     |               |
     ↓               ↓
Personalized     Cold-Start
Recommendations  Recommendations
     |               |
     +-------+-------+
             ↓
         Flask API
```

---

## ✨ Features

- Synthetic user-item interaction dataset generation
- Exploratory Data Analysis (EDA)
- User and item encoding
- User-item interaction matrix
- Collaborative filtering using Truncated SVD
- Personalized recommendations for existing users
- Cold-start recommendations for new users
- Recommendation ranking
- Previously interacted item filtering
- RMSE evaluation
- Precision@10 evaluation
- Recall@10 evaluation
- User activity monitoring
- Item activity monitoring
- Rating distribution monitoring
- Basic data drift detection
- Flask REST API
- Git and GitHub version control

---

# 📊 Dataset

A synthetic dataset was generated to simulate user-item interactions.

### Dataset Statistics

| Metric | Value |
|---|---:|
| Total interactions | 29,101 |
| Unique users | 1,000 |
| Unique items | 500 |
| Average rating | 3.61 |
| Minimum interactions per user | 10 |
| Maximum interactions per user | 45 |
| Minimum interactions per item | 34 |
| Maximum interactions per item | 81 |

### Dataset Columns

```text
user_id
item_id
rating
timestamp
```

### Rating Distribution

| Rating | Number of Interactions |
|---:|---:|
| 1 | 2,324 |
| 2 | 3,515 |
| 3 | 5,865 |
| 4 | 8,750 |
| 5 | 8,647 |

---

# 🔍 Exploratory Data Analysis

The project includes an exploratory analysis script to understand the interaction dataset.

The analysis checks:

- Dataset shape
- Column information
- Missing values
- Unique users
- Unique items
- Interactions per user
- Interactions per item
- Average rating
- Rating distribution
- Popular items

### Dataset Shape

```text
29101 rows × 4 columns
```

### Missing Values

```text
No missing values
```

### User Activity

```text
Average interactions per user: 29.10
Minimum interactions per user: 10
Maximum interactions per user: 45
```

### Item Activity

```text
Average interactions per item: 58.20
Minimum interactions per item: 34
Maximum interactions per item: 81
```

---

# 🤖 Machine Learning Approach

The recommendation system uses **collaborative filtering with Truncated Singular Value Decomposition (SVD)**.

## Training Pipeline

```text
Interaction Dataset
        ↓
Data Preprocessing
        ↓
User Encoding
        ↓
Item Encoding
        ↓
User-Item Matrix
        ↓
Truncated SVD
        ↓
Latent Feature Representation
        ↓
Predicted User-Item Scores
        ↓
Recommendation Ranking
```

## Model Configuration

| Component | Configuration |
|---|---|
| Algorithm | Truncated SVD |
| Latent dimensions | 30 |
| User encoding | LabelEncoder |
| Item encoding | LabelEncoder |
| Train/Test split | 80/20 |

### Training Data

```text
Training interactions: 23,280
Testing interactions:   5,821
User-Item matrix:       1000 × 500
Latent dimensions:      30
```

---

# 🎯 Personalized Recommendations

For an existing user, the recommendation engine:

1. Identifies the user.
2. Retrieves the user's predicted item scores.
3. Finds items that the user has already interacted with.
4. Removes those items from the recommendation candidates.
5. Sorts the remaining items by predicted score.
6. Returns the top-N recommendations.

## Example

For:

```text
User ID: 103
```

The system generated:

| Rank | Item ID | Predicted Score |
|---:|---:|---:|
| 1 | 293 | 1.151 |
| 2 | 329 | 1.070 |
| 3 | 278 | 1.055 |
| 4 | 169 | 0.943 |
| 5 | 18 | 0.931 |
| 6 | 395 | 0.887 |
| 7 | 174 | 0.876 |
| 8 | 179 | 0.867 |
| 9 | 364 | 0.864 |
| 10 | 311 | 0.854 |

---

# 🆕 Cold-Start Recommendation

## Problem

A new user has no previous interaction history.

Because collaborative filtering depends on user interaction history, the system needs an alternative recommendation strategy.

## Solution

The project implements a popularity-based cold-start strategy.

The recommendation score considers:

- Average item rating
- Number of ratings
- Global average rating

This helps avoid recommending an item only because it has a small number of high ratings.

## Example

For a new user:

```text
User ID: 99999
```

The system generated:

```text
405
267
446
110
72
293
239
315
465
67
```

The API identifies this user as:

```text
type: cold_start
```

---

# 📈 Model Evaluation

The recommendation engine was evaluated using:

- RMSE
- Precision@10
- Recall@10

## Evaluation Results

```text
RMSE:          3.2291
Precision@10:  0.0009
Recall@10:     0.0030
Users evaluated: 965
```

These results represent the baseline performance of the current model on the synthetic dataset.

The evaluation metrics can be improved in future versions by using more advanced recommendation algorithms and ranking techniques.

---

# 📡 Monitoring

A monitoring module was developed to track important characteristics of the recommendation data.

## Current Monitoring Results

```text
Total interactions:           29,101
Unique users:                 1,000
Unique items:                 500

Average rating:               3.614

Average interactions/user:    29.10
Median interactions/user:     29
Maximum interactions/user:    45

Average interactions/item:    58.20
Median interactions/item:     58
Maximum interactions/item:    81

Inactive users (<3):          0
Low activity items (<5):      0
```

---

# 📊 Rating Distribution Monitoring

Current rating distribution:

| Rating | Percentage |
|---:|---:|
| 1 | 7.99% |
| 2 | 12.08% |
| 3 | 20.15% |
| 4 | 30.07% |
| 5 | 29.71% |

This monitoring can help identify changes in user behavior over time.

---

# 📉 Data Drift Detection

The monitoring system compares baseline and recent rating averages.

```text
Baseline average rating: 3.611
Recent average rating:   3.618
Drift:                    0.007
```

### Result

```text
No significant drift detected.
```

The drift monitoring can be extended in future versions with statistical tests and automated alerts.

---

# 🌐 REST API

The recommendation engine is exposed through a Flask REST API.

## Start the API

```bash
python src/api.py
```

The API runs at:

```text
http://127.0.0.1:5000
```

---

# 🔗 API Endpoints

## 1. API Status

```http
GET /
```

Example:

```json
{
  "message": "Recommendation Engine API is running",
  "endpoints": [
    "/recommend?user_id=103",
    "/recommend?user_id=99999"
  ]
}
```

---

## 2. Personalized Recommendation

```http
GET /recommend?user_id=103
```

Example response:

```json
{
  "recommendations": [
    {
      "item_id": 293,
      "predicted_rating": 1.151
    },
    {
      "item_id": 329,
      "predicted_rating": 1.07
    },
    {
      "item_id": 278,
      "predicted_rating": 1.055
    }
  ],
  "type": "personalized",
  "user_id": 103
}
```

---

## 3. Cold-Start Recommendation

```http
GET /recommend?user_id=99999
```

Example response:

```json
{
  "recommendations": [
    405,
    267,
    446,
    110,
    72,
    293,
    239,
    315,
    465,
    67
  ],
  "type": "cold_start",
  "user_id": 99999
}
```

---

## 4. Custom Number of Recommendations

The API supports a `top_n` parameter.

Example:

```http
GET /recommend?user_id=103&top_n=5
```

This returns the top 5 recommendations.

The API accepts values from 1 to 50.

---

# 🗂️ Project Structure

```text
Recommendation-Engine/
│
├── data/
│   └── interactions.csv
│
├── models/
│   ├── recommendation_model.joblib
│   ├── user_encoder.joblib
│   ├── item_encoder.joblib
│   ├── user_item_matrix.joblib
│   └── predicted_matrix.joblib
│
├── outputs/
│
├── notebooks/
│
├── src/
│   ├── create_dataset.py
│   ├── explore_data.py
│   ├── train_model.py
│   ├── recommend.py
│   ├── cold_start.py
│   ├── evaluate_model.py
│   ├── monitoring.py
│   └── api.py
│
├── .gitignore
├── README.md
├── requirements.txt
└── venv/
```

---

# 🛠️ Technologies Used

### Programming Language

- Python

### Data Science and Machine Learning

- Pandas
- NumPy
- Scikit-learn
- Truncated SVD
- Joblib

### API

- Flask

### Data Visualization

- Matplotlib
- Seaborn

### Development Tools

- VS Code
- Jupyter
- Git
- GitHub
- PowerShell

---

# ⚙️ Installation

## Clone the Repository

```bash
git clone https://github.com/sonu-balagavi15/Recommendation-Engine.git
```

Navigate to the project:

```bash
cd Recommendation-Engine
```

---

# 🐍 Create Virtual Environment

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

---

# 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Project

## Step 1 — Generate Dataset

```bash
python src/create_dataset.py
```

---

## Step 2 — Explore Dataset

```bash
python src/explore_data.py
```

---

## Step 3 — Train Recommendation Model

```bash
python src/train_model.py
```

---

## Step 4 — Generate Personalized Recommendations

```bash
python src/recommend.py
```

---

## Step 5 — Test Cold-Start Recommendations

```bash
python src/cold_start.py
```

---

## Step 6 — Evaluate Model

```bash
python src/evaluate_model.py
```

---

## Step 7 — Run Monitoring

```bash
python src/monitoring.py
```

---

## Step 8 — Start Flask API

```bash
python src/api.py
```

Open the API in your browser:

```text
http://127.0.0.1:5000
```

---

# 🧪 API Testing

The API was tested with both existing and new users.

### Existing User

```text
User ID: 103
Type: personalized
```

### New User

```text
User ID: 99999
Type: cold_start
```

Both recommendation paths were successfully tested through the Flask API.

---

# 💾 Generated Model Files

The training process generates the following files:

```text
recommendation_model.joblib
user_encoder.joblib
item_encoder.joblib
user_item_matrix.joblib
predicted_matrix.joblib
```

These files allow the trained recommendation system to be loaded without retraining the model every time the API starts.

---

# 🔮 Future Improvements

The current system can be extended with:

- Better recommendation ranking
- Precision@5 and Recall@5
- NDCG@K
- Recommendation diversity
- Recommendation novelty
- Neural collaborative filtering
- Matrix factorization improvements
- Real-time monitoring dashboard
- Automated drift alerts
- API authentication
- Docker containerization
- Cloud deployment
- Frontend recommendation dashboard
- Real-time recommendation serving
- A/B testing of recommendation strategies

---

# 🎓 Learning Outcomes

Through this project, the following concepts were implemented:

- Recommendation systems
- Collaborative filtering
- Dimensionality reduction
- Truncated SVD
- User-item matrices
- Cold-start problem
- Model evaluation
- Recommendation ranking
- Data monitoring
- Data drift detection
- REST API development
- Model serialization
- Git and GitHub workflow

---

# 👨‍💻 Author

## Sonu Balagavi

B.E. Computer Science Engineering  
AGM Rural College of Engineering and Technology

### GitHub

https://github.com/sonu-balagavi15

### LinkedIn

https://www.linkedin.com/in/sonu-balagavi

### Portfolio

https://my-portfolio-nine-chi-72.vercel.app/

---

# 📌 Project Repository

https://github.com/sonu-balagavi15/Recommendation-Engine

---

## ⭐ Project Summary

This project demonstrates an end-to-end machine learning recommendation pipeline covering:

```text
Data Generation
      ↓
Data Exploration
      ↓
Model Training
      ↓
Personalized Recommendations
      ↓
Cold-Start Recommendations
      ↓
Model Evaluation
      ↓
Monitoring & Drift Detection
      ↓
Flask REST API
      ↓
GitHub
```

The project combines machine learning, recommendation systems, data monitoring, and backend API development into a single end-to-end application.
