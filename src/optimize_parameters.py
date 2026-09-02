import pandas as pd


def save_results(results, output_path):

    rows = []

    for model, metrics in results.items():
        rows.append({
            "Model": model,
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "R2 Score": metrics["R2"]
        })

    df = pd.DataFrame(rows)

    df.to_csv(output_path, index=False)

    return df