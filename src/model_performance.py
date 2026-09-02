import pandas as pd
import matplotlib.pyplot as plt
import os


def model_performance():

    print("Model Performance Analysis Started...")

    # Create output folders
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/metrics", exist_ok=True)


    # Model results (from train_model.py output)
    metrics = {
        "Model": [
            "Linear Regression",
            "Random Forest"
        ],

        "MAE": [
            0.25257431955740195,
            0.0828725615099185
        ],

        "RMSE": [
            0.414792714699582,
            0.1416067656634649
        ],

        "R2": [
            0.8646314784839153,
            0.9842230353416
        ]
    }


    df = pd.DataFrame(metrics)


    print("\nModel Performance:")
    print(df)


    # Save metrics CSV
    df.to_csv(
        "results/metrics/model_metrics.csv",
        index=False
    )


    # -------------------------
    # MAE Comparison
    # -------------------------

    plt.figure(figsize=(8,5))

    plt.bar(
        df["Model"],
        df["MAE"]
    )

    plt.xlabel("Model")
    plt.ylabel("MAE")
    plt.title("Model Comparison - MAE")

    plt.savefig(
        "results/figures/model_MAE_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()



    # -------------------------
    # RMSE Comparison
    # -------------------------

    plt.figure(figsize=(8,5))

    plt.bar(
        df["Model"],
        df["RMSE"]
    )

    plt.xlabel("Model")
    plt.ylabel("RMSE")
    plt.title("Model Comparison - RMSE")

    plt.savefig(
        "results/figures/model_RMSE_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()



    # -------------------------
    # R2 Comparison
    # -------------------------

    plt.figure(figsize=(8,5))

    plt.bar(
        df["Model"],
        df["R2"]
    )

    plt.xlabel("Model")
    plt.ylabel("R² Score")
    plt.title("Model Comparison - R²")

    plt.savefig(
        "results/figures/model_R2_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()



    print("\nModel Performance Analysis Completed Successfully")
    print("Saved:")
    print("results/metrics/model_metrics.csv")
    print("results/figures/model_MAE_comparison.png")
    print("results/figures/model_RMSE_comparison.png")
    print("results/figures/model_R2_comparison.png")



if __name__ == "__main__":
    model_performance()