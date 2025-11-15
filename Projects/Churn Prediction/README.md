# 🎮 Player Churn Prediction Using Behavioral Log Data

A Machine Learning Project for Customer Retention in Gaming

## 📌 Project Overview

This project focuses on predicting player churn based on mobile game activity logs.
Using real gameplay data (timestamps, scores, session behavior), the goal is to identify players who are likely to stop playing so that game developers can intervene with targeted retention strategies.

This project demonstrates:
- Data cleaning and preprocessing
- Feature engineering based on academic churn modeling practices
- Observation Period (OP) & Churn Prediction Period (CP) framework
- Handling highly imbalanced datasets
- Training and evaluating ML models
- Model explainability using SHAP

## 🧠 Key Objectives
- Define churn using OP (activity window) and CP (inactivity window).
- Engineer meaningful behavioral features from the OP only.
- Handle severe class imbalance in churn labels.
- Train multiple machine learning models and compare performance.
- Explain model predictions using SHAP on Random Forest and XGBoost.

## 📂 Dataset
The dataset contains anonymized gameplay logs with:
    - device (player ID)
    - time (Unix timestamp)
    - score
    - Gameplay date/time
Each row represents a game session.

##🔧 Methodology
### 1. Data Cleaning
- Convert Unix timestamps to datetime
- Remove blank or missing device IDs
- Extract date, hour, weekday, month
- Handle skewed score distribution

### 2. Define OP & CP Windows
Academic game-churn research often uses:
- Short OP (3–7 days) to capture early engagement
- Short-to-mid CP (7–30 days) to detect inactivity
Duration used in this project:
- OP = 3 days
- CP = 15 days

Churn logic:
- Played in OP → YES
- Played 0 sessions in CP → Churner (1)
- Played ≥1 session in CP → Non-churner (0)

### 3. Feature Engineering (10 important OP-based features)
- Total games played
- Total score
- Average score
- Max score
- Score variability (std)
- Median score
- Number of active days
- Mean time between sessions
- Maximum games per day
- Score trend (slope over time)

### 4. Handling Imbalance
- Original churn distribution ≈ 87% churn, 13% non-churn
- Tried class weights
- Tried SMOTE oversampling
- SMOTE provided much better balanced performance

### 5. Modeling
Models tested:
- Logistic Regression
- Random Forest
- XGBoost
- Gradient Boosting

### 6. Explainability
- SHAP summary plots used to understand:
- Which features contribute most to churn prediction
- How features influence the model positively or negatively

## 📊 Results 
- Random Forest is the best-performing model with an accuracy of 0.87 and perfectly balanced precision/recall.
- SMOTE significantly improved model performance for all classifiers.
- Ensemble models (Random Forest, XGBoost) clearly outperformed linear models.
