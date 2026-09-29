from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, cross_validate

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

DATA_PATH = (
    BASE
    / "data"
    / "processed"
    / "master_machining_dataset.csv"
)

RESULT_PATH = (
    BASE
    / "results"
    / "metrics"
    / "group_cross_validation.csv"
)

RESULT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

# --------------------------------------------------
# Load dataset
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

X = df[FEATURES]
y = df[TARGET]

# --------------------------------------------------
# Create machining-condition groups
# --------------------------------------------------

groups = (
    df[FEATURES]
    .astype(str)
    .agg("|".join, axis=1)
)

print("Total rows:", len(df))
print(
    "Unique machining conditions:",
    groups.nunique()
)

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
        )
    ],
    remainder="passthrough",
)

# --------------------------------------------------
# Models
# --------------------------------------------------

models = {
    "Linear Regression":
        LinearRegression(),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            random_state=42,
        ),

    "Extra Trees":
        ExtraTreesRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),
}

# --------------------------------------------------
# 5-Fold Group Cross-Validation
# --------------------------------------------------

cv = GroupKFold(
    n_splits=5
)

scoring = {
    "R2": "r2",
    "MAE": "neg_mean_absolute_error",
    "RMSE": "neg_root_mean_squared_error",
}

results = []

print(
    "\n5-FOLD GROUP CROSS-VALIDATION\n"
)

for name, model in models.items():

    pipeline = Pipeline(
        steps=[
            (
                "preprocessing",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    scores = cross_validate(
        pipeline,
        X,
        y,
        groups=groups,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )

    r2_scores = scores["test_R2"]

    mae_scores = (
        -scores["test_MAE"]
    )

    rmse_scores = (
        -scores["test_RMSE"]
    )

    result = {
        "Model": name,

        "Mean R2":
            np.mean(r2_scores),

        "R2 SD":
            np.std(r2_scores),

        "Mean MAE":
            np.mean(mae_scores),

        "MAE SD":
            np.std(mae_scores),

        "Mean RMSE":
            np.mean(rmse_scores),

        "RMSE SD":
            np.std(rmse_scores),
    }

    results.append(result)

    print(name)

    print(
        "R2 folds:",
        np.round(
            r2_scores,
            4
        )
    )

    print(
        "Mean R2:",
        round(
            np.mean(r2_scores),
            4
        ),
        "+/-",
        round(
            np.std(r2_scores),
            4
        ),
    )

    print(
        "Mean MAE:",
        round(
            np.mean(mae_scores),
            4
        )
    )

    print(
        "Mean RMSE:",
        round(
            np.mean(rmse_scores),
            4
        )
    )

    print("-" * 50)

# --------------------------------------------------
# Final comparison
# --------------------------------------------------

results_df = pd.DataFrame(
    results
)

results_df = (
    results_df
    .sort_values(
        "Mean R2",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)

print(
    "\nFINAL GROUP-CV COMPARISON\n"
)

print(
    results_df.to_string(
        index=False
    )
)

# --------------------------------------------------
# Best model
# --------------------------------------------------

best = results_df.iloc[0]

print(
    "\nBest model:",
    best["Model"]
)

print(
    "Mean R2:",
    round(
        best["Mean R2"],
        4
    )
)

print(
    "R2 SD:",
    round(
        best["R2 SD"],
        4
    )
)

print(
    "Mean MAE:",
    round(
        best["Mean MAE"],
        4
    )
)

print(
    "Mean RMSE:",
    round(
        best["Mean RMSE"],
        4
    )
)

# --------------------------------------------------
# Save results
# --------------------------------------------------

results_df.to_csv(
    RESULT_PATH,
    index=False,
)

print(
    "\nResults saved to:"
)

print(
    RESULT_PATH
)

print(
    "\nValidation method:"
)

print(
    "5-fold GroupKFold."
)

print(
    "Identical machining parameter "
    "combinations remain within the "
    "same fold to reduce leakage."
)