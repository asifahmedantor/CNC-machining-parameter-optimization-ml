# CNC Machining Parameter Optimization Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Regression-green)
![Model](https://img.shields.io/badge/Best%20Model-Random%20Forest-orange)


## Project Overview

This project focuses on developing a machine learning based system for predicting CNC machining surface roughness (Ra) and optimizing machining parameters.

The objective is to identify the best machining conditions that produce minimum surface roughness and improve machining quality.

A Random Forest Regression model is developed and compared with Linear Regression for accurate prediction.


---

# Objectives

- Predict CNC machining surface roughness (Ra)
- Analyze the effect of machining parameters
- Compare machine learning models
- Identify important machining parameters
- Optimize machining conditions


---

# Dataset Description

The dataset contains experimental CNC machining parameters.

## Input Parameters

| Parameter | Description |
|---|---|
| Depth_of_Cut_ap | Depth of cut |
| Feed_Rate_f | Feed rate |
| Cutting_Speed_vc | Cutting speed |
| Material | Workpiece material |
| Tool | Cutting tool |


## Target Variable

```
Surface_Roughness_Ra
```


---

# Methodology

The complete workflow:

```
Data Collection

↓

Data Preprocessing

↓

Exploratory Data Analysis

↓

Machine Learning Model Training

↓

Model Evaluation

↓

Parameter Optimization

↓

Feature Importance Analysis

↓

Final Report Generation
```


---

# Machine Learning Models

## Linear Regression

Used as a baseline regression model.


## Random Forest Regression

Used for learning complex nonlinear relationships between machining parameters and surface roughness.


---

# Model Performance


| Model | MAE | RMSE | R² Score |
|---|---|---|---|
| Linear Regression | 0.2525 | 0.4147 | 0.8646 |
| Random Forest | 0.0828 | 0.1416 | 0.9842 |


## Best Model

**Random Forest Regression**

Reason:

- Lowest prediction error
- Highest R² score
- Better prediction capability


---

# Feature Importance Analysis


| Parameter | Importance |
|---|---|
| Feed Rate (f) | 95.8% |
| Material | 2.39% |
| Cutting Speed (vc) | 0.96% |
| Depth of Cut (ap) | 0.82% |
| Tool | ~0% |


The analysis indicates that feed rate is the most influential factor affecting surface roughness.


---

# Optimization Result


The optimized machining parameters:


| Parameter | Optimal Value |
|---|---|
| Depth of Cut | 0.5 |
| Feed Rate | 0.1 |
| Cutting Speed | 300 |
| Material | 2 |
| Tool | 0 |


Predicted Surface Roughness:

```
Ra = 0.3735
```


---

# Project Structure

```
CNC-machining-parameter-optimization-ml/

│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── optimize_parameters.py
│   ├── visualize_results.py
│   ├── feature_analysis.py
│   ├── model_performance.py
│   └── generate_report.py
│
├── results/
│   ├── figures/
│   ├── metrics/
│   └── report/
│
├── requirements.txt
└── README.md
```


---

# Installation


Create environment:

```bash
conda create -n cnc_ml python=3.12
```


Activate:

```bash
conda activate cnc_ml
```


Install dependencies:

```bash
pip install -r requirements.txt
```


---

# Run Project


### Data Processing

```bash
python -m src.data_preprocessing
```


### Train Model

```bash
python -m src.train_model
```


### Evaluate Model

```bash
python -m src.evaluate_model
```


### Optimize Parameters

```bash
python -m src.optimize_parameters
```


### Generate Visualizations

```bash
python -m src.visualize_results
```


### Feature Analysis

```bash
python -m src.feature_analysis
```


---

# Results

The Random Forest model achieved:

- R² Score: **0.9842**
- MAE: **0.0828**
- RMSE: **0.1416**


The developed system successfully predicts surface roughness and identifies optimized CNC machining parameters.


---

# Conclusion

Machine learning can effectively improve CNC machining parameter selection.

The proposed Random Forest based approach provides accurate surface roughness prediction and reduces the need for excessive experimental trials.


---

# Future Work

- Integration with real-time CNC systems
- Testing advanced models such as XGBoost and Neural Networks
- Web-based prediction application
- Real-time machining optimization


---

# Author

CNC Machining Parameter Optimization ML Project
