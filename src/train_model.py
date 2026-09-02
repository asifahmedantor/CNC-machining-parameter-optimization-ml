import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def train_models(df, target_column):

    results = {}

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]


    # Remove unnecessary columns
    X = X.drop(
        columns=[
            "Sample_ID",
            "Experiment_Path"
        ],
        errors="ignore"
    )


    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


    # Models
    models = {

        "Linear Regression":
            LinearRegression(),

        "Random Forest":
            RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )
    }


    # Create result folder
    os.makedirs(
        "results",
        exist_ok=True
    )


    for name, model in models.items():

        print(f"Training {name}...")


        # Train model
        model.fit(
            X_train,
            y_train
        )


        # Prediction
        prediction = model.predict(
            X_test
        )


        # Metrics
        mae = mean_absolute_error(
            y_test,
            prediction
        )


        rmse = mean_squared_error(
            y_test,
            prediction
        ) ** 0.5


        r2 = r2_score(
            y_test,
            prediction
        )


        results[name] = {

            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        }


        # Save Random Forest model
        if name == "Random Forest":

            joblib.dump(
                model,
                "results/random_forest_model.pkl"
            )

            print(
                "Random Forest model saved successfully"
            )


    return results



if __name__ == "__main__":


    # Load processed data

    df = pd.read_csv(
        "data/processed/processed_data.csv"
    )


    # Target column

    target_column = "Surface_Roughness_Ra"


    # Train

    results = train_models(
        df,
        target_column
    )


    print(
        "\nTraining completed successfully"
    )


    print(
        results
    )