import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os


def feature_analysis():

    print("Feature analysis started...")


    # Load trained Random Forest model
    model = joblib.load(
        "results/random_forest_model.pkl"
    )


    # Feature names (same order used during training)
    features = [
        "Depth_of_Cut_ap",
        "Feed_Rate_f",
        "Cutting_Speed_vc",
        "Material",
        "Tool"
    ]


    # Get feature importance
    importance = model.feature_importances_


    # Create dataframe
    importance_df = pd.DataFrame(
        {
            "Feature": features,
            "Importance": importance
        }
    )


    # Sort descending
    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )


    print("\nFeature Importance:")
    print(importance_df)


    # Create output folder
    os.makedirs(
        "results/figures",
        exist_ok=True
    )


    # Save importance values
    importance_df.to_csv(
        "results/feature_importance.csv",
        index=False
    )


    # Plot
    plt.figure(figsize=(8,5))


    plt.bar(
        importance_df["Feature"],
        importance_df["Importance"]
    )


    plt.xlabel(
        "Machining Parameters"
    )


    plt.ylabel(
        "Importance Score"
    )


    plt.title(
        "Random Forest Feature Importance"
    )


    plt.xticks(
        rotation=45
    )


    plt.tight_layout()


    plt.savefig(
        "results/figures/feature_importance.png"
    )


    plt.close()


    print(
        "\nFeature importance saved successfully"
    )

    print(
        "Saved: results/figures/feature_importance.png"
    )



if __name__ == "__main__":

    feature_analysis()