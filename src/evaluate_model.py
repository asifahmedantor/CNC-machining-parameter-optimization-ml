import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model():

    # Load processed data
    df = pd.read_csv(
        "data/processed/processed_data.csv"
    )

    target_column = "Surface_Roughness_Ra"

    # Split features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]


    # Remove unnecessary columns
    X = X.drop(
        columns=["Sample_ID", "Experiment_Path"],
        errors="ignore"
    )


    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


    # Load trained model
    model = joblib.load(
        "results/random_forest_model.pkl"
    )


    # Prediction
    prediction = model.predict(X_test)


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


    print("Model Evaluation Results")
    print("------------------------")
    print(f"MAE  : {mae}")
    print(f"RMSE : {rmse}")
    print(f"R2   : {r2}")



if __name__ == "__main__":
    evaluate_model()