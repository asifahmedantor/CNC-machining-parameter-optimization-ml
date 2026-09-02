import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_models(df, target_column):

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
    }

    results = {}

    for name, model in models.items():

        model.fit(X_train, y_train)

        prediction = model.predict(X_test)

        results[name] = {
            "MAE": mean_absolute_error(y_test, prediction),
            "RMSE": mean_squared_error(
                y_test, prediction, squared=False
            ),
            "R2": r2_score(y_test, prediction)
        }

    return results