import pandas as pd
import joblib
import itertools
import os


def optimize_parameters():

    print("Optimization started...")


    # Load processed data
    df = pd.read_csv(
        "data/processed/processed_data.csv"
    )


    # Load trained Random Forest model
    model = joblib.load(
        "results/random_forest_model.pkl"
    )


    print("Model loaded successfully")


    # Parameter values
    depth_values = df["Depth_of_Cut_ap"].unique()

    feed_values = df["Feed_Rate_f"].unique()

    speed_values = df["Cutting_Speed_vc"].unique()

    material_values = df["Material"].unique()

    tool_values = df["Tool"].unique()



    # Limit combinations for fast optimization

    depth_values = depth_values[:20]

    feed_values = feed_values[:20]

    speed_values = speed_values[:20]

    material_values = material_values[:5]

    tool_values = tool_values[:5]



    best_result = None

    count = 0



    # Test combinations

    for values in itertools.product(
        depth_values,
        feed_values,
        speed_values,
        material_values,
        tool_values
    ):


        count += 1


        depth, feed, speed, material, tool = values



        input_data = pd.DataFrame(

            [[
                depth,
                feed,
                speed,
                material,
                tool
            ]],

            columns=[

                "Depth_of_Cut_ap",

                "Feed_Rate_f",

                "Cutting_Speed_vc",

                "Material",

                "Tool"
            ]
        )


        prediction = model.predict(
            input_data
        )[0]



        result = {

            "Depth_of_Cut_ap": depth,

            "Feed_Rate_f": feed,

            "Cutting_Speed_vc": speed,

            "Material": material,

            "Tool": tool,

            "Predicted_Surface_Roughness": prediction

        }



        if (
            best_result is None
            or prediction < best_result["Predicted_Surface_Roughness"]
        ):

            best_result = result



    print(
        f"Total combinations tested: {count}"
    )


    print(
        "\nOptimization completed successfully"
    )


    print(
        "--------------------------------"
    )


    print(
        best_result
    )



    # Save result

    os.makedirs(
        "results",
        exist_ok=True
    )


    pd.DataFrame(
        [best_result]
    ).to_csv(

        "results/optimized_parameters.csv",

        index=False
    )


    print(
        "\nSaved: results/optimized_parameters.csv"
    )




if __name__ == "__main__":

    optimize_parameters()