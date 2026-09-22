Heavy Metal Pollution -- HPI Prediction

1. Project Goal

The aim of this project is to use heavy-metal concentration values
to predict the Heavy Pollution Index (HPI) using Machine Learning.

2. Dataset

The notebook uses two Excel files:

Heavy_Metal_Concentrations.xlsx

Pollution_Indices.xls

Main metal features used:

Zn, Cd, Pb, Cu, Ni, Mn, As, Cr

Target:

HPI

3. Complete Process

Excel Dataset
     ↓
Load Data
     ↓
Clean Column Names
     ↓
Create Date from Year + Season
     ↓
Check Missing Values
     ↓
Remove Duplicates
     ↓
Merge the Two Datasets
     ↓
EDA + Correlation
     ↓
Select Heavy-Metal Features
     ↓
Set HPI as Target
     ↓
Train/Test Split (80/20)
     ↓
Train ML Models
     ↓
Evaluate Models
     ↓
Feature Importance
     ↓
Save Model
     ↓
Predict HPI for New Data

4. Data Preprocessing

First, the Excel files are loaded using Pandas.

Column names are cleaned by removing extra spaces.

Season information such as winter, spring, summer and autumn is
converted into a date.

Missing values and duplicate rows are checked and removed.

Then both datasets are merged using:

Date

Site

5. Exploratory Data Analysis

The merged data is analyzed using:

Basic statistics

Heavy-metal distributions

Correlation with HPI

Correlation matrix

This helps understand the relationship between metal concentrations and
HPI.

6. Feature and Target

Input Features

Zn
Cd
Pb
Cu
Ni
Mn
As
Cr

Target

HPI

So the model learns:

Heavy-metal concentrations → HPI

7. Train/Test Split

The dataset is divided into:

80% Training data

20% Testing data

random_state=42 is used so the result can be reproduced.

8. Models Used

Four regression models are trained:

Linear Regression

Decision Tree Regressor

Random Forest Regressor

XGBoost Regressor

9. Model Evaluation

The models are evaluated using:

MAE

Average prediction error.

Lower is better.

RMSE

Measures prediction error and gives more importance to large errors.

Lower is better.

R²

Shows how well the model explains the target variation.

Higher is better.

10. Feature Importance

Random Forest feature importance is calculated to understand which heavy
metals contribute most to the model's HPI predictions.

11. Prediction Analysis

The notebook compares:

Actual HPI
vs
Predicted HPI

It also plots prediction errors to understand model performance.

12. Save the Model

The trained Random Forest model is saved as:

heavy_metal_hpi_random_forest.pkl

The feature list is saved as:

hpi_features.pkl

These files can later be loaded without training the model again.

13. New Prediction

A new sample containing the eight metal concentrations is given to the
saved model.

The model returns:

Predicted HPI

14. How to Run

Install required libraries:

pip install pandas numpy matplotlib scikit-learn xgboost openpyxl xlrd joblib

Open:

HPI.ipynb

Keep the Excel datasets in the correct folder and run the notebook
from top to bottom.

15. Important Note

The current notebook uses a small dataset, so the model results should
be treated as a baseline.

For the final project, use a larger real-world dataset and proper
cross-validation/independent testing before making strong claims about
real-world performance.

16. Final Output

The project produces:

Clean merged dataset

EDA and correlation plots

Four trained ML models

MAE, RMSE and R² comparison

Random Forest feature importance

Actual vs predicted HPI plot

Prediction error analysis

Saved Random Forest model

HPI prediction for new samples
