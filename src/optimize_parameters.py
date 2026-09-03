import numpy as np
import pandas as pd
import joblib


print("Optimization started...")

# Load trained model
model = joblib.load("models/random_forest.pkl")

print("Model loaded successfully")


def optimize_parameters():

    best_ra = float("inf")
    best_result = None


    # Parameter range
    depth_values = np.linspace(0.2, 1.0, 5)
    feed_values = np.linspace(0.05, 0.25, 5)
    speed_values = np.linspace(100, 400, 10)


    # Encoded values
    materials = {
        "20MnCr5": 0,
        "EN AW-6082": 1,
        "41Cr4": 2
    }


    tools = {
        "Tool1": 0
    }


    results = []


    for depth in depth_values:

        for feed in feed_values:

            for speed in speed_values:

                for material_name, material_code in materials.items():

                    for tool_name, tool_code in tools.items():


                        input_data = pd.DataFrame({

                            "Sample_ID":[0],

                            "Experiment_Path":[0],

                            "Depth_of_Cut_ap":[depth],

                            "Feed_Rate_f":[feed],

                            "Cutting_Speed_vc":[speed],

                            "Material":[material_code],

                            "Tool":[tool_code]

                        })


                        prediction = model.predict(input_data)[0]


                        results.append({

                            "Depth_of_Cut":round(depth,3),

                            "Feed_Rate":round(feed,3),

                            "Cutting_Speed":round(speed,2),

                            "Material":material_name,

                            "Tool":tool_name,

                            "Predicted_Ra":round(prediction,4)

                        })


                        if prediction < best_ra:

                            best_ra = prediction

                            best_result = results[-1]



    # Save result
    result_df = pd.DataFrame(results)

    result_df.to_csv(
        "results/optimized_parameters.csv",
        index=False
    )


    return best_result



# Run optimization

result = optimize_parameters()


print("\nBest Machining Parameters")
print("-------------------------")

for key,value in result.items():

    print(key,":",value)