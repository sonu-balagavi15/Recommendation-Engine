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
````

---

# ⭐ Rating Distribution

The dataset contains ratings from 1 to 5.

| Rating | Percentage |
| -----: | ---------: |
|      1 |      7.99% |
|      2 |     12.08% |
|      3 |     20.15% |
|      4 |     30.07% |
|      5 |     29.71% |

The average rating is approximately:

```text
3.614
```

---

# 🔍 Data Exploration

The project includes a complete exploratory data analysis stage.

The exploration script checks:

* Dataset shape
* Column names
* Missing values
* Number of users
* Number of items
* User interaction frequency
* Item interaction frequency
* Rating distribution
* Average rating
* Popular items

Run:

```bash
python src/explore_data.py
```

---

# 🧠 Machine Learning Approach

The recommendation system uses **collaborative filtering**.

The main model uses:

```text
User-Item Interaction Matrix
            ↓
        Truncated SVD
            ↓
      Latent Features
            ↓
   Predicted Rating Matrix
            ↓
   Ranked Recommendations
```

---

# ⚙️ Model Training

The training process performs the following steps:

1. Load interaction data.
2. Encode users and items.
3. Split the dataset into training and testing data.
4. Create a user-item interaction matrix.
5. Apply Truncated SVD.
6. Generate latent representations.
7. Reconstruct the predicted rating matrix.
8. Save the trained model and supporting files.

### Training Dataset

```text
Training interactions: 23,280
Testing interactions: 5,821
```

### User-Item Matrix

```text
Users: 1,000
Items: 500
Matrix size: 1000 × 500
```

### Latent Dimensions

```text
30
```

Run model training:

```bash
python src/train_model.py
```

---

# 📁 Saved Model Files

The trained model generates the following files:

```text
models/
│
├── recommendation_model.joblib
├── user_encoder.joblib
├── item_encoder.joblib
├── user_item_matrix.joblib
└── predicted_matrix.joblib
```

### Description

| File                          | Purpose                                    |
| ----------------------------- | ------------------------------------------ |
| `recommendation_model.joblib` | Trained SVD recommendation model           |
| `user_encoder.joblib`         | Encodes user IDs                           |
| `item_encoder.joblib`         | Encodes item IDs                           |
| `user_item_matrix.joblib`     | User-item interaction matrix               |
| `predicted_matrix.joblib`     | Predicted ratings used for recommendations |

---

# 🎯 Personalized Recommendations

For an existing user, the recommendation engine:

1. Finds the user's encoded index.
2. Retrieves predicted item scores.
3. Identifies items the user has already interacted with.
4. Removes those items.
5. Sorts remaining items by predicted score.
6. Returns the top recommendations.

Example:

```bash
python src/recommend.py
```

Example user:

```text
User ID: 103
```

The system generates personalized recommendations for the user.

Example output:

```text
Item 293
Item 329
Item 278
Item 169
Item 18
Item 395
Item 174
Item 179
Item 364
Item 311
```

---

# 🆕 Cold-Start Recommendation

A major problem in recommendation systems is the **cold-start problem**.

A new user has no previous interaction history, so a personalized recommendation model cannot immediately understand their preferences.

This project solves the problem using a popularity-based recommendation strategy.

The system calculates:

* Average rating
* Number of ratings
* Weighted score

It then recommends highly rated items with sufficient interaction history.

Run:

```bash
python src/cold_start.py
```

Example new user:

```text
User ID: 99999
```

Example recommendations:

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

This allows the system to provide recommendations even when the user has no historical data.

---

# 📈 Model Evaluation

The recommendation engine is evaluated using:

* RMSE
* Precision@10
* Recall@10

Run:

```bash
python src/evaluate_model.py
```

### Evaluation Results

```text
RMSE:          3.2291
Precision@10:  0.0009
Recall@10:     0.0030
Users evaluated: 965
```

### Metrics

#### RMSE

Root Mean Squared Error measures the difference between actual ratings and predicted ratings.

Lower RMSE indicates smaller prediction error.

#### Precision@10

Precision@10 measures how many of the top 10 recommended items are relevant to the user's test interactions.

#### Recall@10

Recall@10 measures how many relevant test items were successfully included in the top 10 recommendations.

---

# 📊 Model Monitoring

Machine learning systems need monitoring after deployment.

This project includes a monitoring script that checks:

* Total interactions
* Unique users
* Unique items
* Average rating
* Rating distribution
* User activity
* Item activity
* Baseline average rating
* Recent average rating
* Data drift

Run:

```bash
python src/monitoring.py
```

---

# 🔄 Data Drift Detection

The monitoring system compares historical/baseline data with recent data.

Example result:

```text
Baseline average rating: 3.611
Recent average rating:   3.618
Drift:                    0.007
```

Result:

```text
No significant drift detected.
```

The monitoring component can help identify changes in incoming data that may affect recommendation quality.

---

# 📡 REST API

The recommendation system is exposed using a Flask REST API.

The API supports:

* Health/status endpoint
* Personalized recommendations
* Cold-start recommendations
* Configurable number of recommendations
* Input validation

---

# 🌐 Live API

The application is deployed on Render.

Live URL:

[https://recommendation-engine-h5gc.onrender.com](https://recommendation-engine-h5gc.onrender.com)

---

# 🔗 API Endpoints

## 1. Home Endpoint

```text
GET /
```

Live:

[https://recommendation-engine-h5gc.onrender.com/](https://recommendation-engine-h5gc.onrender.com/)

Example response:

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

```text
GET /recommend?user_id=103
```

Live:

[https://recommendation-engine-h5gc.onrender.com/recommend?user_id=103](https://recommendation-engine-h5gc.onrender.com/recommend?user_id=103)

This endpoint returns personalized recommendations for an existing user.

Example:

```json
{
  "user_id": 103,
  "type": "personalized",
  "recommendations": [
    {
      "item_id": 293,
      "predicted_rating": 1.151
    },
    {
      "item_id": 329,
      "predicted_rating": 1.070
    }
  ]
}
```

---

## 3. Cold-Start Recommendation

```text
GET /recommend?user_id=99999
```

Live:

[https://recommendation-engine-h5gc.onrender.com/recommend?user_id=99999](https://recommendation-engine-h5gc.onrender.com/recommend?user_id=99999)

Because user `99999` does not exist in the training data, the API automatically uses the cold-start strategy.

Example response:

```json
{
  "user_id": 99999,
  "type": "cold_start",
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
  ]
}
```

---

# 🔢 Custom Number of Recommendations

The API supports the `top_n` parameter.

Example:

```text
/recommend?user_id=103&top_n=5
```

This returns the top 5 recommendations.

The API accepts:

```text
top_n = 1 to 50
```

---

# 🛡️ API Validation

The API validates incoming requests.

If `user_id` is missing:

```json
{
  "error": "user_id is required"
}
```

If `top_n` is outside the valid range:

```json
{
  "error": "top_n must be between 1 and 50"
}
```

---

# 🚀 Deployment

The application is deployed using **Render**.

### Build Command

```bash
pip install -r requirements.txt
```

### Start Command

```bash
gunicorn src.api:app
```

The Flask application runs through Gunicorn in the production environment.

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
├── notebooks/
│
├── outputs/
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
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🧩 Project Workflow

The complete workflow is:

```text
Dataset Generation
        ↓
Data Exploration
        ↓
Data Preprocessing
        ↓
User/Item Encoding
        ↓
Train/Test Split
        ↓
User-Item Matrix
        ↓
Truncated SVD
        ↓
Predicted Rating Matrix
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
        ↓
Render Deployment
```

---

# 🛠️ Technologies Used

### Programming Language

* Python

### Machine Learning

* Scikit-learn
* Truncated SVD
* Collaborative Filtering

### Data Processing

* Pandas
* NumPy

### Model Serialization

* Joblib

### Backend

* Flask
* Gunicorn

### Deployment

* Render

### Version Control

* Git
* GitHub

---

# 📦 Requirements

The project uses the following Python packages:

```text
Flask
gunicorn
pandas
numpy
scikit-learn
joblib
```

---

# 💻 Local Installation

## Step 1: Clone the Repository

```bash
git clone https://github.com/sonu-balagavi15/Recommendation-Engine.git
```

Go into the project directory:

```bash
cd Recommendation-Engine
```

---

## Step 2: Create Virtual Environment

Windows:

```bash
python -m venv venv
```

Activate:

```bash
venv\Scripts\activate
```

---

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 4: Generate Dataset

```bash
python src/create_dataset.py
```

---

## Step 5: Explore Dataset

```bash
python src/explore_data.py
```

---

## Step 6: Train Model

```bash
python src/train_model.py
```

---

## Step 7: Test Personalized Recommendations

```bash
python src/recommend.py
```

---

## Step 8: Test Cold-Start Recommendations

```bash
python src/cold_start.py
```

---

## Step 9: Evaluate Model

```bash
python src/evaluate_model.py
```

---

## Step 10: Run Monitoring

```bash
python src/monitoring.py
```

---

## Step 11: Start Flask API

```bash
python src/api.py
```

The local API will run at:

```text
http://127.0.0.1:5000
```

---

# 🧪 Testing the Local API

Open:

```text
http://127.0.0.1:5000/
```

For personalized recommendations:

```text
http://127.0.0.1:5000/recommend?user_id=103
```

For cold-start recommendations:

```text
http://127.0.0.1:5000/recommend?user_id=99999
```

---

# 🔐 Cold-Start Logic

The recommendation API automatically determines whether a user is known.

```text
                User ID
                   ↓
          Is user in dataset?
             /          \
           Yes           No
            ↓             ↓
     Personalized     Popularity
     Recommendation   Recommendation
```

This makes the API capable of handling both existing and new users.

---

# 📊 Monitoring Features

The monitoring component checks the health of the recommendation data.

### User Activity

The system calculates:

* Average interactions per user
* Median interactions per user
* Maximum interactions per user
* Inactive users

### Item Activity

The system calculates:

* Average interactions per item
* Median interactions per item
* Maximum interactions per item
* Low-activity items

### Rating Monitoring

The system calculates:

* Average rating
* Rating distribution
* Baseline rating average
* Recent rating average
* Rating drift

---

# 💡 Key Features

* ✅ Machine learning recommendation system
* ✅ Collaborative filtering
* ✅ Truncated SVD
* ✅ Personalized recommendations
* ✅ Cold-start handling
* ✅ Popularity-based fallback
* ✅ RMSE evaluation
* ✅ Precision@10
* ✅ Recall@10
* ✅ User activity monitoring
* ✅ Item activity monitoring
* ✅ Rating distribution monitoring
* ✅ Data drift detection
* ✅ Flask REST API
* ✅ Input validation
* ✅ Production deployment
* ✅ GitHub version control

---

# 📈 Project Results

The project successfully demonstrates a complete recommendation system pipeline.

### Dataset

```text
29,101 interactions
1,000 users
500 items
```

### Model

```text
Truncated SVD
30 latent dimensions
```

### Evaluation

```text
RMSE:          3.2291
Precision@10:  0.0009
Recall@10:     0.0030
```

### Monitoring

```text
Baseline rating: 3.611
Recent rating:   3.618
Drift:           0.007
```

### Deployment

```text
Status: Live
Platform: Render
```

---

# 🔮 Future Improvements

The current system can be extended with:

* Hybrid recommendation systems
* Content-based filtering
* Neural collaborative filtering
* Matrix factorization optimization
* Better hyperparameter tuning
* Improved ranking algorithms
* User profile features
* Item metadata
* Real-time recommendation updates
* Advanced drift detection
* Recommendation explanation
* A web-based user interface
* Authentication
* Recommendation feedback tracking
* A/B testing
* Model retraining pipelines
* Docker deployment
* CI/CD automation

---

# 🎓 Learning Outcomes

Through this project, the following concepts were implemented:

* Data preprocessing
* Exploratory data analysis
* User-item matrices
* Collaborative filtering
* Dimensionality reduction
* Truncated SVD
* Recommendation ranking
* Cold-start problem
* Model evaluation
* Precision and Recall
* RMSE
* Model monitoring
* Data drift detection
* REST API development
* Flask deployment
* Gunicorn
* Git and GitHub
* Cloud deployment using Render

---

# 📚 Project Use Case

This recommendation engine can be adapted for applications such as:

* E-commerce product recommendations
* Movie recommendations
* Music recommendations
* Online course recommendations
* Book recommendations
* News recommendations
* Content recommendation platforms

The current implementation uses generic users and items so that the recommendation pipeline can be adapted to different domains.

---

# 👨‍💻 Author

**Sonu Parashuram Balagavi**

B.E. Computer Science and Engineering
AGM Rural College of Engineering and Technology
2023–2027

---

# 🔗 Project Links

### GitHub

[https://github.com/sonu-balagavi15/Recommendation-Engine](https://github.com/sonu-balagavi15/Recommendation-Engine)

### Live API

[https://recommendation-engine-h5gc.onrender.com](https://recommendation-engine-h5gc.onrender.com)

### Personalized Recommendation API

[https://recommendation-engine-h5gc.onrender.com/recommend?user_id=103](https://recommendation-engine-h5gc.onrender.com/recommend?user_id=103)

### Cold-Start Recommendation API

[https://recommendation-engine-h5gc.onrender.com/recommend?user_id=99999](https://recommendation-engine-h5gc.onrender.com/recommend?user_id=99999)

---

# ⭐ Conclusion

This project implements an end-to-end machine learning recommendation system with personalized recommendations, cold-start handling, evaluation, monitoring, drift detection, REST API integration, and cloud deployment.

The system demonstrates how a recommendation model can be developed from raw interaction data, evaluated using machine learning metrics, monitored for changes in incoming data, exposed through an API, and deployed as a live service.

The project provides a foundation that can be extended into a production-grade recommendation platform using larger datasets, hybrid recommendation methods, real-time feedback, advanced ranking algorithms, and automated model retraining.
