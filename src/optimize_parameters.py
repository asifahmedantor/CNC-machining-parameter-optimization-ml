import numpy as np
import pandas as pd
import joblib


print("Optimization started...")

# Load Extra Trees model
model = joblib.load("models/extra_trees.pkl")

print("Extra Trees model loaded successfully")


def optimize_parameters():

    best_ra = float("inf")
    best_result = None

    # Parameter ranges restricted to the available dataset
    depth_values = np.linspace(0.25, 1.50, 6)
    feed_values = np.linspace(0.10, 0.30, 5)
    speed_values = np.linspace(100, 300, 9)

    # Encoded categorical values
    materials = {
        "20MnCr5": 0,
        "EN AW-6082": 1,
        "41Cr4": 2
    }

    # Only one tool is represented in the current dataset
    tools = {
        "DNMG150608": 0
    }

    results = []

    for depth in depth_values:
        for feed in feed_values:
            for speed in speed_values:
                for material_name, material_code in materials.items():
                    for tool_name, tool_code in tools.items():

                        input_data = pd.DataFrame({
                            "Sample_ID": [0],
                            "Experiment_Path": [0],
                            "Depth_of_Cut_ap": [depth],
                            "Feed_Rate_f": [feed],
                            "Cutting_Speed_vc": [speed],
                            "Material": [material_code],
                            "Tool": [tool_code]
                        })

                        # Match the feature structure used during training
                        input_data = input_data.reindex(
                            columns=model.feature_names_in_,
                            fill_value=0
                        )

                        prediction = model.predict(input_data)[0]

                        result = {
                            "Depth_of_Cut": round(depth, 3),
                            "Feed_Rate": round(feed, 3),
                            "Cutting_Speed": round(speed, 2),
                            "Material": material_name,
                            "Tool": tool_name,
                            "Predicted_Ra": round(prediction, 4)
                        }

                        results.append(result)

                        if prediction < best_ra:
                            best_ra = prediction
                            best_result = result

    # Save all evaluated combinations
    result_df = pd.DataFrame(results)

    result_df.to_csv(
        "results/optimized_parameters.csv",
        index=False
    )

    return best_result


# Run optimization
result = optimize_parameters()

print("\nBest Model-Predicted Machining Parameters")
print("-----------------------------------------")

for key, value in result.items():
    print(key, ":", value)