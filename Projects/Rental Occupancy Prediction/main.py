# Import libraries
import json
import zipfile
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

# 1. LOAD DATA
with open('C:/Users/Anh Thu/Downloads/Master/Machine Learning/ML Challenge/train.json', 'r') as f:
    train_data = json.load(f)
with open('C:/Users/Anh Thu/Downloads/Master/Machine Learning/ML Challenge/test.json', 'r') as f:
    test_data = json.load(f)

train = pd.DataFrame(train_data)
test = pd.DataFrame(test_data)

# 2. MISSING VALUE IMPUTATION

def impute_missing_values_safe(train_df, test_df):
    """Safe imputation - no leakage"""
    train = train_df.copy()
    test = test_df.copy()
    
    # Bathrooms
    train['bathrooms'] = train.groupby('room_type')['bathrooms'].transform(lambda x: x.fillna(x.median()))
    test['bathrooms'] = test.groupby('room_type')['bathrooms'].transform(lambda x: x.fillna(x.median()))
    
    # Cancellation
    if train['cancellation'].isnull().any():
        mode_val = train['cancellation'].mode()[0]
        train['cancellation'] = train['cancellation'].fillna(mode_val)
        test['cancellation'] = test['cancellation'].fillna(mode_val)
    
    # Beds
    median_beds_train = train.groupby('room_type')['beds'].median()
    
    for idx, row in train[train['beds'].isnull()].iterrows():
        if pd.notna(row['rooms']):
            train.at[idx, 'beds'] = max(1, round(row['rooms']))
        else:
            train.at[idx, 'beds'] = median_beds_train.get(row['room_type'], 1)
    
    for idx, row in test[test['beds'].isnull()].iterrows():
        if pd.notna(row['rooms']):
            test.at[idx, 'beds'] = max(1, round(row['rooms']))
        else:
            test.at[idx, 'beds'] = median_beds_train.get(row['room_type'], 1)
    
    # KNN imputation - fit on train, transform both
    features_for_knn = ['rooms', 'bathrooms', 'beds', 'guests', 'lat', 'lon', 'min_nights', 'num_reviews']
    imputer = KNNImputer(n_neighbors=5, weights='distance')
    train[features_for_knn] = imputer.fit_transform(train[features_for_knn])
    test[features_for_knn] = imputer.transform(test[features_for_knn])
    
    train['host'] = train['host'].fillna('unknown_host')
    test['host'] = test['host'].fillna('unknown_host')
    train['name'] = train['name'].fillna('')
    test['name'] = test['name'].fillna('')
    
    return train, test

train, test = impute_missing_values_safe(train, test)

# 3. AMENITIES PARSING
def parse_key_amenities(df):
    df = df.copy()

    df['has_wifi'] = df['facilities'].str.contains('wifi|internet|wi-fi', case=False, na=False).astype(int)
    df['has_kitchen'] = df['facilities'].str.contains('kitchen', case=False, na=False).astype(int)
    df['has_parking'] = df['facilities'].str.contains('parking|garage', case=False, na=False).astype(int)
    df['has_ac'] = df['facilities'].str.contains('air conditioning|ac|cooling', case=False, na=False).astype(int)
    df['has_washer'] = df['facilities'].str.contains('washer|washing machine|laundry', case=False, na=False).astype(int)
    df['has_pool'] = df['facilities'].str.contains('pool|swimming', case=False, na=False).astype(int)
    df['has_gym'] = df['facilities'].str.contains('gym|fitness', case=False, na=False).astype(int)

    df['essential_amenities'] = (df['has_wifi'] + df['has_kitchen'] +
                                 df['has_parking'] + df['has_ac'] + df['has_washer'])
    return df

train = parse_key_amenities(train)
test = parse_key_amenities(test)

# 4. TRAINING STATISTICS
TRAIN_CENTER_LAT = train['lat'].median()
TRAIN_CENTER_LON = train['lon'].median()
TRAIN_HOST_FREQ = train['host'].value_counts().to_dict()

TRAIN_HOST_AVG_RATING = train.groupby('host')['rating'].mean().to_dict()
TRAIN_HOST_AVG_REVIEWS = train.groupby('host')['num_reviews'].mean().to_dict()
TRAIN_HOST_MEDIAN_RATING = train.groupby('host')['rating'].median().to_dict()
TRAIN_HOST_STD_RATING = train.groupby('host')['rating'].std().fillna(0).to_dict()

GLOBAL_AVG_RATING = train['rating'].mean()
GLOBAL_AVG_REVIEWS = train['num_reviews'].mean()

# 5. TARGET ENCODING
def target_encode_with_cv(train_df, test_df, column, target_col='occupancy',
                          n_folds=5, alpha=10):
    global_mean = train_df[target_col].mean()
    train_encoded = np.zeros(len(train_df))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(train_df):
        stats = train_df.iloc[train_idx].groupby(column).agg(
            count=(target_col, 'count'),
            mean=(target_col, 'mean')
        )
        stats['smoothed'] = (stats['mean'] * stats['count'] +
                             global_mean * alpha) / (stats['count'] + alpha)

        train_encoded[val_idx] = train_df.iloc[val_idx][column].map(
            stats['smoothed']).fillna(global_mean)

    stats_full = train_df.groupby(column).agg(
        count=(target_col, 'count'),
        mean=(target_col, 'mean')
    )
    stats_full['smoothed'] = (stats_full['mean'] * stats_full['count'] +
                              global_mean * alpha) / (stats_full['count'] + alpha)
    test_encoded = test_df[column].map(stats_full['smoothed']).fillna(global_mean)

    return train_encoded, test_encoded


def frequency_encode(train_df, test_df, column):
    freq = train_df[column].value_counts(normalize=True).to_dict()
    train_encoded = train_df[column].map(freq).fillna(0)
    test_encoded = test_df[column].map(freq).fillna(0)
    return train_encoded, test_encoded


def count_encode(train_df, test_df, column):
    counts = train_df[column].value_counts().to_dict()
    train_encoded = train_df[column].map(counts).fillna(0)
    test_encoded = test_df[column].map(counts).fillna(0)
    return train_encoded, test_encoded

train['host_target'], test['host_target'] = target_encode_with_cv(train, test, 'host', alpha=20)
train['room_type_target'], test['room_type_target'] = target_encode_with_cv(train, test, 'room_type')
train['listing_type_target'], test['listing_type_target'] = target_encode_with_cv(train, test, 'listing_type')
train['cancellation_target'], test['cancellation_target'] = target_encode_with_cv(train, test, 'cancellation')

train['host_freq'], test['host_freq'] = frequency_encode(train, test, 'host')
train['host_count'], test['host_count'] = count_encode(train, test, 'host')

# 6. LOCATION FEATURES + GEO BINS
train['lat_bin'], lat_bins = pd.qcut(train['lat'], q=20, labels=False, duplicates='drop', retbins=True)
test['lat_bin'] = pd.cut(test['lat'], bins=lat_bins, labels=False)

train['lon_bin'], lon_bins = pd.qcut(train['lon'], q=20, labels=False, duplicates='drop', retbins=True)
test['lon_bin'] = pd.cut(test['lon'], bins=lon_bins, labels=False)

train['lat_bin_target'], test['lat_bin_target'] = target_encode_with_cv(train, test, 'lat_bin')
train['lon_bin_target'], test['lon_bin_target'] = target_encode_with_cv(train, test, 'lon_bin')

train['geo_zone'] = train['lat_bin'].astype(str) + "_" + train['lon_bin'].astype(str)
test['geo_zone'] = test['lat_bin'].astype(str) + "_" + test['lon_bin'].astype(str)

train['geo_zone_target'], test['geo_zone_target'] = target_encode_with_cv(train, test, 'geo_zone')

# 7. FEATURE ENGINEERING
def engineer_features(df, center_lat, center_lon,
                      host_freq_dict, host_avg_rating, host_avg_reviews,
                      host_median_rating, host_std_rating):
    df = df.copy()

    # Review-based features
    df['weighted_reviews'] = df['rating'] * np.log1p(df['num_reviews'])
    df['review_density'] = df['num_reviews'] / (df['min_nights'] + 1)
    df['review_credibility'] = df['rating'] * np.sqrt(df['num_reviews'])
    df['is_highly_rated'] = (df['rating'] >= 4.8).astype(int)
    df['rating_squared'] = df['rating'] ** 2

    # Distance
    df['distance_from_center'] = np.sqrt(
        (df['lat'] - center_lat) ** 2 + (df['lon'] - center_lon) ** 2
    )

    # Capacity-related
    df['total_capacity'] = df['guests'] + df['beds'] * 0.5

    # Host features
    df['host_listings'] = df['host'].map(host_freq_dict).fillna(1)

    # Amenities
    df['num_amenities'] = df['facilities'].str.split(',').str.len()

    # Host statistical features
    df['host_avg_rating'] = df['host'].map(host_avg_rating).fillna(GLOBAL_AVG_RATING)
    df['host_avg_reviews'] = df['host'].map(host_avg_reviews).fillna(GLOBAL_AVG_REVIEWS)
    df['host_median_rating'] = df['host'].map(host_median_rating).fillna(GLOBAL_AVG_RATING)
    df['host_rating_std'] = df['host'].map(host_std_rating).fillna(0)

    return df

train_fe = engineer_features(train, TRAIN_CENTER_LAT, TRAIN_CENTER_LON,
                             TRAIN_HOST_FREQ, TRAIN_HOST_AVG_RATING,
                             TRAIN_HOST_AVG_REVIEWS, TRAIN_HOST_MEDIAN_RATING,
                             TRAIN_HOST_STD_RATING)

test_fe = engineer_features(test, TRAIN_CENTER_LAT, TRAIN_CENTER_LON,
                            TRAIN_HOST_FREQ, TRAIN_HOST_AVG_RATING,
                            TRAIN_HOST_AVG_REVIEWS, TRAIN_HOST_MEDIAN_RATING,
                            TRAIN_HOST_STD_RATING)
print("Feature engineering completed.")
print(f"Train features shape: {train_fe.shape}")
print(f"Test features shape: {test_fe.shape}")

train_only = set(train_fe.columns) - set(test_fe.columns)
test_only = set(test_fe.columns) - set(train_fe.columns)


# 8. FINAL FEATURES
y = train_fe['occupancy'].values
y_transformed = np.log1p(y)

drop_cols = [
    'occupancy', 'host', 'name', 'facilities',
    'lat_bin', 'lon_bin', 'geo_zone',
    'room_type', 'listing_type', 'cancellation'
]

feature_cols = [c for c in train_fe.columns if c not in drop_cols]

X = train_fe[feature_cols].values
X_test = test_fe[feature_cols].values

# 9. FEATURE SELECTION
X_df = pd.DataFrame(X, columns=feature_cols)
X_test_df = pd.DataFrame(X_test, columns=feature_cols)

variance_threshold = VarianceThreshold(threshold=0.01)
X_var = variance_threshold.fit_transform(X_df)
selected_var_features = X_df.columns[variance_threshold.get_support()]
X_df = X_df[selected_var_features]
X_test_df = X_test_df[selected_var_features]

# Drop highly correlated features
corr_matrix = X_df.corr().abs()
features_to_drop = set()

for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        if corr_matrix.iloc[i, j] > 0.95:
            f1, f2 = corr_matrix.columns[i], corr_matrix.columns[j]
            c1 = abs(np.corrcoef(X_df[f1], y_transformed)[0, 1])
            c2 = abs(np.corrcoef(X_df[f2], y_transformed)[0, 1])
            features_to_drop.add(f1 if c1 < c2 else f2)

X_df = X_df.drop(columns=features_to_drop)
X_test_df = X_test_df.drop(columns=features_to_drop)

X_selected = X_df.values
X_test_selected = X_test_df.values
print(f"Selected features shape: {X_selected.shape}")

# 10. OUTLIER CAPPING
X_df_capped = pd.DataFrame(X_selected, columns=X_df.columns)
X_test_df_capped = pd.DataFrame(X_test_selected, columns=X_df.columns)

outlier_sensitive = [
    'num_reviews', 'min_nights', 'weighted_reviews',
    'review_density', 'num_amenities', 'host_listings',
    'distance_from_center'
]

for col in outlier_sensitive:
    if col in X_df_capped.columns:
        lb = X_df_capped[col].quantile(0.01)
        ub = X_df_capped[col].quantile(0.99)
        X_df_capped[col] = X_df_capped[col].clip(lb, ub)
        X_test_df_capped[col] = X_test_df_capped[col].clip(lb, ub) 



X_selected = X_df_capped.values
X_test_selected = X_test_df_capped.values

# 11. MODEL TRAINING + STACKING
kf = KFold(n_splits=10, shuffle=True, random_state=42)
seeds = [42, 123, 456]

models_config = []
for seed in seeds:
    models_config.extend([
        (f'XGB_s{seed}', XGBRegressor(
            n_estimators=993,
            learning_rate=0.015340557833544858,
            max_depth=7,
            min_child_weight=2,
            subsample=0.8263734207131492,
            colsample_bytree=0.7539865169219098,
            gamma=0.012043286828836668,
            reg_alpha=0.12598811264986068,
            reg_lambda=1.0923082494581384,
            random_state=seed,
            n_jobs=-1,
            verbosity=0
        )),
        (f'LGBM_s{seed}', LGBMRegressor(
            n_estimators=658,
            learning_rate=0.029160947514895962,
            max_depth=8,
            num_leaves=48,
            min_child_samples=12,
            subsample=0.7466230720966263,
            colsample_bytree=0.736233864337132,
            reg_alpha=0.003289067895007186,
            reg_lambda=1.2662361026942746,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1
        )),
        (f'Cat_s{seed}', CatBoostRegressor(
            iterations=939,
            learning_rate=0.04756794054666366,
            depth=8,
            l2_leaf_reg=5,
            subsample=0.8494064702566914,
            random_state=seed,
            verbose=0,
            thread_count=-1
        ))
    ])

oof_preds = np.zeros((len(y), len(models_config)))
test_preds = np.zeros((len(X_test_selected), len(models_config)))

for idx, (name, model) in enumerate(models_config):
    fold_test = []
    for tr_idx, val_idx in kf.split(X_selected):
        model.fit(X_selected[tr_idx], y_transformed[tr_idx])
        oof_preds[val_idx, idx] = np.expm1(model.predict(X_selected[val_idx]))
        fold_test.append(np.expm1(model.predict(X_test_selected)))
    test_preds[:, idx] = np.mean(fold_test, axis=0)

# Weighted ensemble based on model MAE
base_maes = [mean_absolute_error(y, oof_preds[:, i]) for i in range(len(models_config))]
weights = 1 / (np.array(base_maes) ** 2)
weights /= weights.sum()

test_weighted = np.average(test_preds, axis=1, weights=weights)
print(f"Ensembled OOF MAE: {mean_absolute_error(y, np.average(oof_preds, axis=1, weights=weights)):.4f}")

# 12. SAVE SUBMISSION FILE
predictions = [{'occupancy': float(p)} for p in test_weighted]

with open('predicted.json', 'w') as f:
    json.dump(predictions, f)

with zipfile.ZipFile('submission.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write('predicted.json')

print("Submission saved as submission.zip")
