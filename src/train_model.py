# ==========================================
# CNC Machining Parameter Optimization
# Random Forest Model Training
# ==========================================


import pandas as pd
import numpy as np
import os
import joblib


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)



# ================================
# Load Dataset
# ================================

data_path = "data/processed/processed_data.csv"

df = pd.read_csv(data_path)


print("Dataset Loaded Successfully")

print(df.head())

print("\nColumns:")
print(df.columns)



# ================================
# Encode Categorical Data
# ================================

encoder = LabelEncoder()


for col in df.select_dtypes(include="object").columns:

    df[col] = encoder.fit_transform(df[col])



# ================================
# Feature and Target
# ================================


target = "Surface_Roughness_Ra"


X = df.drop(
    target,
    axis=1
)


y = df[target]



# ================================
# Train Test Split
# ================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



# ================================
# Linear Regression
# ================================

linear_model = LinearRegression()

linear_model.fit(
    X_train,
    y_train
)


linear_pred = linear_model.predict(
    X_test
)



linear_r2 = r2_score(
    y_test,
    linear_pred
)


linear_mae = mean_absolute_error(
    y_test,
    linear_pred
)


linear_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        linear_pred
    )
)



# ================================
# Random Forest
# ================================

rf_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)


rf_model.fit(
    X_train,
    y_train
)


rf_pred = rf_model.predict(
    X_test
)



rf_r2 = r2_score(
    y_test,
    rf_pred
)


rf_mae = mean_absolute_error(
    y_test,
    rf_pred
)


rf_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        rf_pred
    )
)



# ================================
# Performance Table
# ================================

results = pd.DataFrame(
    {
        "Model":
        [
            "Linear Regression",
            "Random Forest"
        ],

        "R2 Score":
        [
            linear_r2,
            rf_r2
        ],

        "MAE":
        [
            linear_mae,
            rf_mae
        ],

        "RMSE":
        [
            linear_rmse,
            rf_rmse
        ]
    }
)


print("\nModel Performance:")
print(results)



# ================================
# Save Model
# ================================

os.makedirs(
    "models",
    exist_ok=True
)


joblib.dump(
    rf_model,
    "models/random_forest.pkl"
)


print("\nRandom Forest Model Saved!")



# ================================
# Save Metrics
# ================================

os.makedirs(
    "results/metrics",
    exist_ok=True
)


results.to_csv(
    "results/metrics/model_performance.csv",
    index=False
)



print("\nTraining Completed Successfully!")