from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
)

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE = Path(__file__).resolve().parents[1]

DATA_PATH = BASE / "data" / "processed" / "master_machining_dataset.csv"
RESULT_PATH = BASE / "results" / "metrics" / "grouped_validation.csv"

RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

FEATURES = [
    "Depth_of_Cut_ap",
    "Feed_Rate_f",
    "Cutting_Speed_vc",
    "Material",
    "Tool",
]

TARGET = "Surface_Roughness_Ra"

NUMERIC_FEATURES = [
    "Depth_of_Cut_ap",
    "Feed_Rate_f",
    "Cutting_Speed_vc",
]

CATEGORICAL_FEATURES = [
    "Material",
    "Tool",
]

# --------------------------------------------------
# Create machining-condition groups
# --------------------------------------------------
# Rows with exactly the same machining parameters
# receive the same group ID.

groups = (
    df[FEATURES]
    .astype(str)
    .agg("|".join, axis=1)
)

print("Total rows:", len(df))
print("Unique machining conditions:", groups.nunique())

# --------------------------------------------------
# Group-aware train/test split
# --------------------------------------------------

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)

train_idx, test_idx = next(
    splitter.split(
        df[FEATURES],
        df[TARGET],
        groups=groups,
    )
)

X_train = df.iloc[train_idx][FEATURES]
X_test = df.iloc[test_idx][FEATURES]

y_train = df.iloc[train_idx][TARGET]
y_test = df.iloc[test_idx][TARGET]

train_groups = set(groups.iloc[train_idx])
test_groups = set(groups.iloc[test_idx])

overlap = train_groups.intersection(test_groups)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Training conditions:", len(train_groups))
print("Testing conditions:", len(test_groups))
print("Condition overlap:", len(overlap))

# --------------------------------------------------
# Preprocessing
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            CATEGORICAL_FEATURES,
        ),
    ],
    remainder="passthrough",
)

# --------------------------------------------------
# Models
# --------------------------------------------------

models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42,
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        random_state=42,
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=200,
        random_state=42,
    ),
}

# --------------------------------------------------
# Training and evaluation
# --------------------------------------------------

results = []

for name, model in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    results.append(
        {
            "Model": name,
            "R2 Score": r2,
            "MAE": mae,
            "RMSE": rmse,
        }
    )

# --------------------------------------------------
# Results
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "R2 Score",
    ascending=False,
)

print("\nGROUP-AWARE VALIDATION RESULTS\n")
print(results_df.to_string(index=False))

best_model = results_df.iloc[0]

print("\nBest model:", best_model["Model"])
print("R2:", round(best_model["R2 Score"], 4))
print("MAE:", round(best_model["MAE"], 4))
print("RMSE:", round(best_model["RMSE"], 4))

results_df.to_csv(
    RESULT_PATH,
    index=False,
)

print("\nSaved:")
print(RESULT_PATH)

print(
    "\nValidation rule:"
    "\nIdentical machining parameter combinations "
    "cannot appear in both training and test sets."
)