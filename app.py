# ==========================================
# CNC AI OPTIMIZATION SYSTEM
# APP.PY
# PART 1/4
# ==========================================


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os


import plotly.express as px
import plotly.graph_objects as go



# ==========================================
# PAGE CONFIG
# ==========================================


st.set_page_config(

    page_title="CNC AI Optimization System",

    page_icon="⚙️",

    layout="wide"

)



# ==========================================
# HEADER
# ==========================================


st.title(
    "⚙️ CNC Machining Surface Roughness Prediction"
)


st.markdown(

"""
Machine Learning Based Surface Roughness Prediction & Parameter Optimization
"""

)


st.divider()



# ==========================================
# BASE DIRECTORY
# ==========================================


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)



# ==========================================
# LOAD MODEL
# ==========================================


@st.cache_resource
def load_model():

    model_path = os.path.join(

        BASE_DIR,

        "models",

        "best_model.pkl"

    )


    model = joblib.load(
        model_path
    )


    return model



model = load_model()

# ==========================================
# LOAD ENCODER
# ==========================================

@st.cache_resource
def load_encoder():

    encoder_path = os.path.join(
        BASE_DIR,
        "models",
        "encoders.pkl"
    )

    encoder = joblib.load(
        encoder_path
    )

    return encoder


encoder = load_encoder()


# ==========================================
# LOAD METRICS
# ==========================================


@st.cache_data
def load_metrics():

    path = os.path.join(

        BASE_DIR,

        "results",

        "metrics",

        "model_performance.csv"

    )


    if os.path.exists(path):

        return pd.read_csv(path)


    return None



metrics_df = load_metrics()



# ==========================================
# LOAD OPTIMIZATION DATA
# ==========================================


@st.cache_data
def load_optimization():

    path = os.path.join(

        BASE_DIR,

        "results",

        "optimized_parameters.csv"

    )


    if os.path.exists(path):

        return pd.read_csv(path)


    return None



opt_df = load_optimization()
# ==========================================
# LIVE SURFACE ROUGHNESS PREDICTION
# ==========================================


st.header(
    "🔮 Live Surface Roughness Prediction"
)



col1, col2, col3 = st.columns(3)



with col1:

    depth = st.number_input(

        "Depth of Cut (ap)",

        value=0.8

    )



with col2:

    feed = st.number_input(

        "Feed Rate (f)",

        value=0.10

    )



with col3:

    speed = st.number_input(

        "Cutting Speed (vc)",

        value=200.0

    )




material = st.selectbox(
    "Material",
    list(encoder["Material"].classes_)
)

tool = st.selectbox(
    "Tool",
    list(encoder["Tool"].classes_)
)


if st.button(
    "Predict Surface Roughness"
):

    try:

        input_data = pd.DataFrame({

            "Depth_of_Cut_ap": [depth],

            "Feed_Rate_f": [feed],

            "Cutting_Speed_vc": [speed],

            "Material": [material],

            "Tool": [tool],

            "Experiment_Path": [0],

            "Sample_ID": [0]

        })


        # Encode categorical data

        if "Material" in input_data.columns:

            input_data["Material"] = encoder["Material"].transform(
                input_data["Material"]
            )


        if "Tool" in input_data.columns:

            input_data["Tool"] = encoder["Tool"].transform(
                input_data["Tool"]
            )


        # Match training features

        input_data = input_data[
            model.feature_names_in_
        ]


        prediction = model.predict(
            input_data
        )


        st.success(
            f"Predicted Surface Roughness (Ra): {prediction[0]:.4f}"
        )


    except Exception as e:

        st.error(e)



st.divider()


st.divider()
# ==========================================
# MODEL PERFORMANCE COMPARISON
# ==========================================


st.header(
    "📊 Model Performance Comparison"
)



if metrics_df is not None:


    st.dataframe(

        metrics_df,

        use_container_width=True

    )



    best_model = metrics_df.loc[

        metrics_df["R2 Score"].idxmax()

    ]



    c1, c2, c3 = st.columns(3)



    with c1:

        st.metric(

            "🏆 Best Model",

            best_model["Model"]

        )



    with c2:

        st.metric(

            "R² Score",

            round(

                best_model["R2 Score"],

                4

            )

        )



    with c3:

        st.metric(

            "RMSE",

            round(

                best_model["RMSE"],

                4

            )

        )



else:


    st.warning(

        "Model performance file not found"

    )



st.divider()




# ==========================================
# FEATURE IMPORTANCE
# ==========================================


st.header(

    "📈 Feature Importance"

)




feature_file = os.path.join(

    BASE_DIR,

    "results",

    "feature_importance.csv"

)



if os.path.exists(feature_file):


    fi = pd.read_csv(

        feature_file

    )



    fig = px.bar(

        fi,

        x="Feature",

        y="Importance",

        title="Machining Parameter Importance"

    )



    st.plotly_chart(

        fig,

        use_container_width=True

    )



else:


    st.warning(

        "Feature importance file not found"

    )



st.divider()
# ==========================================
# OPTIMIZATION MODULE
# ==========================================


st.header(
    "🔍 Optimization Module"
)



if opt_df is not None:


    st.success(

        "Optimal Parameters Found!"

    )


    best = opt_df.sort_values(

        "Predicted_Ra"

    ).iloc[0]



    st.subheader(

        "🏆 Best Machining Conditions"

    )



    c1, c2, c3, c4 = st.columns(4)



    c1.metric(

        "Depth of Cut",

        best["Depth_of_Cut"]

    )


    c2.metric(

        "Feed Rate",

        best["Feed_Rate"]

    )


    c3.metric(

        "Cutting Speed",

        best["Cutting_Speed"]

    )


    c4.metric(

        "Predicted Ra",

        round(

            best["Predicted_Ra"],

            4

        )

    )



    st.subheader(

        "Top 10 Optimized Machining Conditions"

    )


    st.dataframe(

        opt_df.head(10),

        use_container_width=True

    )



    csv = opt_df.to_csv(

        index=False

    )


    st.download_button(

        "⬇ Download Optimization Results",

        data=csv,

        file_name="CNC_Optimization_Result.csv",

        mime="text/csv"

    )



else:


    st.warning(

        "Optimization file not found"

    )



st.divider()





# ==========================================
# 3D OPTIMIZATION SURFACE
# ==========================================


st.header(

    "🌐 3D Optimization Surface"

)



if opt_df is not None:


    try:


        fig3d = go.Figure(

            data=[

                go.Scatter3d(

                    x=opt_df["Cutting_Speed"],

                    y=opt_df["Feed_Rate"],

                    z=opt_df["Predicted_Ra"],

                    mode="markers",

                    marker=dict(

                        size=6,

                        color=opt_df["Predicted_Ra"],

                        colorscale="Viridis"

                    )

                )

            ]

        )


        fig3d.update_layout(

            title="Cutting Parameters vs Surface Roughness",

            height=700

        )


        st.plotly_chart(

            fig3d,

            use_container_width=True

        )


    except Exception as e:


        st.warning(e)



st.divider()





# ==========================================
# PARAMETER EFFECT ANALYSIS
# ==========================================


st.header(

    "📊 Parameter Effect Analysis"

)



if opt_df is not None:


    col1, col2 = st.columns(2)



    with col1:


        fig1 = px.scatter(

            opt_df,

            x="Feed_Rate",

            y="Predicted_Ra",

            title="Feed Rate vs Surface Roughness"

        )


        st.plotly_chart(

            fig1,

            use_container_width=True

        )



    with col2:


        fig2 = px.scatter(

            opt_df,

            x="Cutting_Speed",

            y="Predicted_Ra",

            title="Cutting Speed vs Surface Roughness"

        )


        st.plotly_chart(

            fig2,

            use_container_width=True

        )



st.divider()




# ==========================================
# COMPLETE REPORT DOWNLOAD
# ==========================================


st.header(

    "📥 Download Complete Report"

)



if opt_df is not None:


    report = opt_df.to_csv(

        index=False

    )


    st.download_button(

        "Download CNC AI Optimization Report",

        data=report,

        file_name="CNC_AI_Optimization_Report.csv",

        mime="text/csv"

    )



st.divider()




# ==========================================
# FOOTER
# ==========================================


st.caption(

    "CNC Machining Parameter Optimization using Machine Learning"

)