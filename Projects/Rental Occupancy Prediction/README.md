# Rental Property Occupancy Prediction 


## Project Overview

This project was developed for a Machine Learning competition at Tilburg University. The goal was to predict the occupancy rate of rental properties based on diverse features like location, amenities, room types, and host metadata.

I achieved a **Top 10% ranking (17/170)** by implementing a robust feature engineering pipeline and a weighted ensemble of Gradient Boosted Decision Trees (GBDTs).

-----

## Key Highlights & Skills

  * **Advanced Feature Engineering:** Extracted geographical bins, parsed complex "facilities" strings, and calculated distance-from-center metrics.
  * **Leakage-Free Preprocessing:** Implemented custom cross-validated target encoding to prevent data leakage.
  * **Missing Value Imputation:** Used a hybrid approach of Group-by-Median and **KNN Imputation** for high-dimensional spatial data.
  * **Ensemble Modeling:** Built a weighted ensemble of **XGBoost, LightGBM, and CatBoost** with K-Fold cross-validation.
  * **Clean Code:** Modularized the pipeline into distinct stages: cleaning, engineering, selection, and training.

-----

## Technical Stack

  * **Language:** Python
  * **Data Manipulation:** `pandas`, `numpy`
  * **Machine Learning:** `scikit-learn`, `XGBoost`, `LightGBM`, `CatBoost`
  * **Techniques:** Target Encoding, Variance Thresholding, Outlier Capping, K-Fold Cross-Validation.

-----

## Methodology

### 1\. Data Cleaning & Imputation

To maintain data integrity, I used a multi-tier imputation strategy:

  * **Contextual Imputation:** Filled bathrooms and beds based on `room_type` medians.
  * **Spatial Imputation:** Used `KNNImputer` for latitude, longitude, and number of reviews to capture local neighborhood trends.

### 2\. Feature Engineering (The "Difference Maker")

I transformed the raw JSON data into high-signal features:

  * **Amenity Parsing:** Boolean flags for high-value features (WiFi, Kitchen, AC, etc.) and an "Essential Amenity Count."
  * **Geospatial Binning:** Created latitude/longitude zones to capture neighborhood-specific occupancy trends.
  * **Host Statistics:** Aggregated host performance metrics (average rating, review density) to capture host-level influence.
  * **Target Encoding:** Used smoothed, CV-based target encoding for high-cardinality categorical variables like `host` and `geo_zone`.

### 3\. Model Architecture

I employed a **10-Fold Cross-Validation** strategy with a **Weighted Ensemble** of three state-of-the-art GBDT models:

  * **XGBoost:** Excellent at capturing complex interactions.
  * **LightGBM:** Fast training and effective leaf-wise growth.
  * **CatBoost:** Naturally handles categorical nuances and reduces overfitting.

**Final Prediction:** The models were combined using weights inversely proportional to their individual MAE (Mean Absolute Error) scores to ensure the most accurate models had the highest influence.

-----

## Results

  * **Leaderboard Rank:** 17 / 170
  * **Mean Absolute Error (MAE):** 0.1386
  * **Validation Strategy:** 10-Fold CV with Log-Transformation of the target variable to handle skewness.

-----

## Project Structure

```text
├── train.json           # Training dataset
├── test.json            # Hidden test set
├── main.py              # Full end-to-end pipeline
├── predicted.json       # Model output
└── README.md            # Project documentation
```

## How to Run

1.  Ensure you have the dependencies installed:
    ```bash
    pip install numpy pandas scikit-learn xgboost lightgbm catboost
    ```
2.  Change the file path in the code and run the script:
    ```bash
    python main.py
    ```

-----

*Note: This project was completed as an individual assignment under Tilburg University's Machine Learning course guidelines.*