import pandas as pd
import matplotlib.pyplot as plt
import os


def visualize_results():

    print("Visualization started...")


    # Load optimized results
    df = pd.read_csv(
        "results/optimized_parameters.csv"
    )


    # Create figure folder
    os.makedirs(
        "results/figures",
        exist_ok=True
    )


    # -------------------------------
    # 1. Top 10 Best Parameters
    # -------------------------------

    top10 = df.head(10)


    plt.figure(figsize=(10,6))

    plt.bar(
        range(len(top10)),
        top10["Predicted_Surface_Roughness"]
    )

    plt.xlabel(
        "Parameter Combination"
    )

    plt.ylabel(
        "Predicted Surface Roughness (Ra)"
    )

    plt.title(
        "Top 10 Optimized Machining Parameters"
    )


    plt.tight_layout()


    plt.savefig(
        "results/figures/top10_optimized_parameters.png"
    )

    plt.close()



    # -------------------------------
    # 2. Depth of Cut vs Roughness
    # -------------------------------


    plt.figure(figsize=(8,5))


    plt.scatter(
        df["Depth_of_Cut_ap"],
        df["Predicted_Surface_Roughness"]
    )


    plt.xlabel(
        "Depth of Cut (ap)"
    )

    plt.ylabel(
        "Predicted Surface Roughness (Ra)"
    )


    plt.title(
        "Effect of Depth of Cut on Surface Roughness"
    )


    plt.tight_layout()


    plt.savefig(
        "results/figures/depth_vs_roughness.png"
    )


    plt.close()



    # -------------------------------
    # 3. Feed Rate vs Roughness
    # -------------------------------


    plt.figure(figsize=(8,5))


    plt.scatter(
        df["Feed_Rate_f"],
        df["Predicted_Surface_Roughness"]
    )


    plt.xlabel(
        "Feed Rate (f)"
    )


    plt.ylabel(
        "Surface Roughness (Ra)"
    )


    plt.title(
        "Effect of Feed Rate on Surface Roughness"
    )


    plt.tight_layout()


    plt.savefig(
        "results/figures/feed_vs_roughness.png"
    )


    plt.close()



    print(
        "Visualization completed successfully"
    )

    print(
        "Saved in results/figures/"
    )



if __name__ == "__main__":

    visualize_results()