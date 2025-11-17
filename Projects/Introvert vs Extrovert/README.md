# Personality Prediction from Social Behavior Data  
Kaggle Machine Learning Challenge  

## 🧠 Project Overview
This project aims to predict a person’s personality type (Introvert or Extrovert) using features related to social behavior, online activity, and personal preferences.  
The model was developed for a Kaggle challenge and focuses on building an end-to-end supervised learning pipeline — from preprocessing to model tuning and evaluation.

The final tuned XGBoost model achieves **96.7% accuracy** on the validation dataset, demonstrating strong predictive performance.

---

## 📂 Dataset
The dataset contains the following key features:

- **Stage_fear** — categorical  
- **Drained_after_socializing** — categorical  
- **Social_event_attendance** — numeric  
- **Time_spent_Alone** — numeric  
- **Going_outside** — numeric  
- **Friends_circle_size** — numeric  
- **Post_frequency** — numeric  
- **Personality** — target label (`Introvert` / `Extrovert`)

Train–test split:  
- 2/3 training data  
- 1/3 validation data  

---

## 🛠️ Methodology

### **1. Preprocessing**
Used a `ColumnTransformer` pipeline:
- One-hot encoding for categorical features  
- StandardScaler for numerical features  
- LabelEncoding for the target variable  

### **2. Baseline Model**
A `DummyClassifier` using the *most frequent* strategy:
- Train accuracy: **0.742**  
- Validation accuracy: **0.734**

This establishes a minimal performance benchmark.

---

## 🚀 Machine Learning Model: XGBoost

### **Hyperparameter Optimization**
Performed **RandomizedSearchCV** over:
- n_estimators  
- max_depth  
- learning_rate  
- subsample  
- colsample_bytree  
- min_child_weight  
- gamma  
- reg_alpha  
- reg_lambda  

Cross-validation: **5-fold**  
Iterations: **50**

### **Best Hyperparameters Found**
subsample: 1.0
reg_lambda: 2
reg_alpha: 0
n_estimators: 250
min_child_weight: 5
max_depth: 4
learning_rate: 0.2
gamma: 0.2
colsample_bytree: 0.6

---

## 📊 Results

| Model                | Train Accuracy | Validation Accuracy |
|---------------------|---------------|---------------------|
| DummyClassifier     | 0.742         | 0.734               |
| XGBoost (tuned)     | **0.971**     | **0.967**           |
