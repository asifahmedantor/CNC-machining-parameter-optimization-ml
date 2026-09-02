# CNC Machining Parameter Optimization using Machine Learning

## Project Overview

This project develops a machine learning model to predict surface roughness (Ra) in CNC machining and identify optimal machining parameters for achieving minimum surface roughness.

## Dataset

- Total Samples: 453
- Input Parameters:
  - Depth of Cut (ap)
  - Feed Rate (f)
  - Cutting Speed (vc)
  - Material
  - Tool

- Target:
  - Surface Roughness (Ra)

## Methodology

The project workflow includes:

- Data preprocessing and cleaning
- Feature encoding for categorical variables
- Exploratory Data Analysis (EDA)
- Machine learning model development
- Model evaluation and comparison
- Optimization of machining parameters for minimum surface roughness


## Machine Learning Models

The following regression models were developed and evaluated:

- Linear Regression
- Random Forest Regression

Random Forest achieved the best prediction performance with higher R² score and lower error values.


## Model Performance

| Model | MAE | RMSE | R² Score |
|---|---|---|---|
| Linear Regression | 0.252574 | 0.414793 | 0.864631 |
| Random Forest | 0.082873 | 0.141607 | 0.984223 |


## Project Structure

```
CNC-machining-parameter-optimization-ml/

├── data/
│   ├── raw/
│   └── processed/

├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_advanced_eda.ipynb

├── results/
│   ├── figures/
│   └── metrics/

├── src/

├── requirements.txt

└── README.md
```


## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## Exploratory Data Analysis

Performed:

- Data cleaning
- Missing value analysis
- Correlation analysis
- Parameter effect visualization
- Outlier detection
- Feature importance analysis

## Model Evaluation

Models Tested:

1. Linear Regression
2. Random Forest Regression

## Best Model Performance

Random Forest achieved the best performance:

- MAE: 0.082873
- RMSE: 0.141607
- R² Score: 0.984223

## Feature Importance

The analysis showed:

1. Feed Rate (f) has the highest influence on surface roughness.
2. Material has moderate influence.
3. Cutting Speed has comparatively lower influence.

## Optimization Result

The optimized machining condition:

- Depth of Cut: 0.805556
- Feed Rate: 0.100
- Cutting Speed: 300
- Predicted Surface Roughness (Ra): 0.442218 µm

## Conclusion

The developed Random Forest model successfully predicts CNC machining surface roughness and identifies optimal machining parameters for improving machining quality.
