import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor
)

from sklearn.linear_model import LinearRegression


# ==============================
# Load Dataset
# ==============================

data_path = "data/processed/processed_data.csv"

df = pd.read_csv(data_path)


# Target
target = "Surface_Roughness_Ra"


X = df.drop(columns=[target])
y = df[target]


# Remove unwanted columns
remove_cols = [
    "Sample_ID",
    "Experiment_Path"
]

for col in remove_cols:
    if col in X.columns:
        X = X.drop(columns=[col])


# Convert categorical columns

X = pd.get_dummies(X)


# ==============================
# Train Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



# ==============================
# Models
# ==============================

models = {

    "Random Forest":
        RandomForestRegressor(
            n_estimators=200,
            random_state=42
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            random_state=42
        ),

    "Extra Trees":
        ExtraTreesRegressor(
            n_estimators=200,
            random_state=42
        ),

    "Linear Regression":
        LinearRegression()

}



results = []


# ==============================
# Training & Evaluation
# ==============================

for name, model in models.items():

    print("\nTraining:", name)

    model.fit(
        X_train,
        y_train
    )


    prediction = model.predict(
        X_test
    )


    r2 = r2_score(
        y_test,
        prediction
    )


    mae = mean_absolute_error(
        y_test,
        prediction
    )


    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            prediction
        )
    )


    results.append({

        "Model": name,
        "R2 Score": round(r2,4),
        "MAE": round(mae,4),
        "RMSE": round(rmse,4)

    })



# ==============================
# Save Results
# ==============================

result_df = pd.DataFrame(results)


os.makedirs(
    "results/metrics",
    exist_ok=True
)


result_df.to_csv(
    "results/metrics/model_comparison.csv",
    index=False
)



# ==============================
# Best Model
# ==============================

best_model_name = (
    result_df
    .sort_values(
        by="R2 Score",
        ascending=False
    )
    .iloc[0]["Model"]
)


print("\n====================")
print("Best Model:", best_model_name)
print("====================")


print(result_df)