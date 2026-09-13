from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
)
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)

BASE = Path(__file__).resolve().parents[1]

FEATURES = [
    "Depth_of_Cut_ap",
    "Feed_Rate_f",
    "Cutting_Speed_vc",
    "Material",
    "Tool",
]
TARGET = "Surface_Roughness_Ra"

# Use original material/tool names, not previously encoded numbers.
df = pd.read_csv(
    BASE / "data/processed/master_machining_dataset.csv"
)

X_train, X_test, y_train, y_test = train_test_split(
    df[FEATURES],
    df[TARGET],
    test_size=0.2,
    random_state=42,
)

X_train = X_train.copy()
X_test = X_test.copy()

encoders = {}
for column in ["Material", "Tool"]:
    encoder = LabelEncoder()
    X_train[column] = encoder.fit_transform(X_train[column])
    X_test[column] = encoder.transform(X_test[column])
    encoders[column] = encoder

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        random_state=42
    ),
    "Extra Trees": ExtraTreesRegressor(
        n_estimators=200, random_state=42
    ),
}

filenames = {
    "Linear Regression": "linear_regression.pkl",
    "Random Forest": "random_forest.pkl",
    "Gradient Boosting": "gradient_boosting.pkl",
    "Extra Trees": "extra_trees.pkl",
}

model_dir = BASE / "models"
metrics_dir = BASE / "results/metrics"
model_dir.mkdir(parents=True, exist_ok=True)
metrics_dir.mkdir(parents=True, exist_ok=True)

rows = []

for name, model in models.items():
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)

    rows.append({
        "Model": name,
        "R2 Score": r2_score(y_test, prediction),
        "MAE": mean_absolute_error(y_test, prediction),
        "RMSE": np.sqrt(mean_squared_error(y_test, prediction)),
    })

    joblib.dump(model, model_dir / filenames[name])

results = pd.DataFrame(rows)
best_name = results.loc[results["R2 Score"].idxmax(), "Model"]
best_model = models[best_name]

joblib.dump(best_model, model_dir / "best_model.pkl")
joblib.dump(encoders, model_dir / "encoders.pkl")

for filename in ["model_performance.csv", "model_comparison.csv"]:
    results.to_csv(metrics_dir / filename, index=False)

importance = (
    best_model.feature_importances_
    if hasattr(best_model, "feature_importances_")
    else np.abs(best_model.coef_)
)

pd.DataFrame({
    "Feature": FEATURES,
    "Importance": importance,
}).to_csv(BASE / "results/feature_importance.csv", index=False)

print(results.to_string(index=False))
print(f"\nBest comparison model: {best_name}")
print("Models and encoders saved successfully.")
print(
    "Note: this original random-row split is exploratory; "
    "repeated settings may appear in both training and test data."
)