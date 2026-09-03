# ⚙️ CNC Machining Parameter Optimization Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Regression-green)
![Framework](https://img.shields.io/badge/Framework-Streamlit-red)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

# 📌 Overview

This project presents an Artificial Intelligence based system for predicting CNC machining surface roughness and optimizing machining parameters using Machine Learning techniques.

The main goal of this research is to develop a data-driven approach that can accurately predict surface roughness (Ra) and determine the optimal machining parameters to improve machining quality, productivity, and efficiency.

The system integrates Machine Learning prediction models with an optimization module to recommend the best machining conditions for achieving minimum surface roughness.

---

# 🎯 Project Objectives

- Predict CNC machining surface roughness (Ra)
- Analyze the influence of machining parameters
- Compare multiple Machine Learning regression models
- Select the best performing prediction model
- Optimize machining parameters for minimum roughness
- Develop an interactive AI-based dashboard

---

# 🚀 System Features


## 1. Live Surface Roughness Prediction

The developed system predicts surface roughness based on CNC machining parameters:


### Input Parameters:

- Depth of Cut (ap)
- Feed Rate (f)
- Cutting Speed (Vc)
- Material Type
- Cutting Tool


### Output:

- Predicted Surface Roughness (Ra)


---

# 🤖 Machine Learning Models


The following regression models are implemented:


| Model | Description |
|---|---|
| Linear Regression | Basic regression model |
| Random Forest | Ensemble learning approach |
| Gradient Boosting | Boosting based regression |
| Extra Trees Regression | High accuracy ensemble model |


---

# 📊 Model Evaluation


Model performance is evaluated using:


- R² Score
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)


## 🏆 Best Performing Model


### Extra Trees Regression


Performance:


```
R² Score : 0.9907

MAE      : 0.0702

RMSE     : 0.1087
```


The model shows excellent prediction accuracy for CNC surface roughness estimation.


---

# 🔍 Optimization Module


The optimization module searches different machining parameter combinations and identifies the best machining conditions for minimum surface roughness.


## Optimization Output:


- Optimal Depth of Cut
- Optimal Feed Rate
- Optimal Cutting Speed
- Material Selection
- Tool Selection
- Predicted Surface Roughness


Example:


```
Optimal Machining Condition


Depth of Cut     : 0.80 mm

Feed Rate        : 0.05 mm/rev

Cutting Speed    : 366.67 m/min

Material         : 41Cr4

Tool             : Tool1

Predicted Ra     : 0.459 μm

```


---

# 📈 Visualization Dashboard


The developed Streamlit dashboard provides:


## Model Performance Analysis

✔ Model comparison  
✔ R² Score comparison  
✔ RMSE comparison  
✔ MAE comparison  


## Feature Analysis

✔ Feature importance visualization  


## Optimization Visualization

✔ 3D Optimization Surface  
✔ Parameter Effect Analysis  
✔ Optimized Parameter Table  


## Reporting System

✔ Downloadable optimization report  


---

# 🏗️ System Workflow


```
CNC Machining Dataset

          ↓

Data Preprocessing

          ↓

Feature Engineering

          ↓

Machine Learning Training

          ↓

Model Evaluation

          ↓

Optimization Algorithm

          ↓

AI Prediction Dashboard

          ↓

Optimal Machining Parameters

```


---

# 📂 Project Structure


```
CNC-machining-parameter-optimization-ml

│
├── app.py
│
├── database.h5
│
├── requirements.txt
│
├── README.md
│
│
├── data
│
│
├── models
│   ├── random_forest.pkl
│   ├── extra_trees.pkl
│
│
├── src
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── optimize_parameters.py
│   ├── feature_analysis.py
│   ├── generate_report.py
│
│
├── results
│   ├── metrics
│   ├── figures
│   └── report
│

```


---

# 🛠️ Technologies Used


## Programming Language

- Python 3.12


## Machine Learning

- Scikit-learn
- Random Forest
- Gradient Boosting
- Extra Trees Regression


## Data Processing

- Pandas
- NumPy


## Visualization

- Plotly
- Matplotlib


## Application Development

- Streamlit


---

# ⚙️ Installation


Clone the repository:


```bash
git clone https://github.com/asifahmedantor/CNC-machining-parameter-optimization-ml.git
```


Navigate into project folder:


```bash
cd CNC-machining-parameter-optimization-ml
```


Install dependencies:


```bash
pip install -r requirements.txt
```


---

# ▶️ Run Application


Start Streamlit application:


```bash
streamlit run app.py
```


Application will open:


```
http://localhost:8501
```


---

# 📸 Dashboard Modules


The application contains:


✅ Live Prediction System

✅ Machine Learning Model Comparison

✅ Feature Importance Dashboard

✅ Optimization Module

✅ Best Machining Condition Finder

✅ 3D Optimization Visualization

✅ Parameter Effect Analysis

✅ Report Generation


---

# 🔮 Future Improvements


Future development areas:


- Real-time CNC machine data integration
- IoT based machining monitoring
- Deep Learning based prediction
- Automated CNC parameter control
- Cloud-based AI machining platform


---

# 👨‍💻 Author


## CNC AI Optimization System


Machine Learning Based CNC Machining Parameter Optimization Project


---

# 📜 License


This project is developed for academic and research purposes.