# Master Thesis: Predicting and Explaining Happiness with Machine Learning and Causal Inference

This repository contains the code and data used for my master thesis on the prediction and explanation of happiness using data from the European Quality of Life Survey (EQLS) 2011.

The project combines machine learning, SHAP-based model interpretation, and causal inference with DoWhy. The main goal is to identify relevant predictors of happiness and compare them with psychological research on subjective well-being.

## Project Overview

The analysis focuses on the EQLS survey item:

> How happy are you?

This variable is used as the target variable. Predictors include health, financial conditions, social relationships, autonomy, meaning in life, trust, and other variables from the EQLS dataset.

The project follows three main steps:

1. Prediction of happiness using machine learning models
2. Interpretation of model predictions using SHAP values
3. Estimation of possible causal effects using DoWhy

## Data

The repository contains the raw and preprocessed EQLS data.

Main preprocessing steps include:

- filtering relevant survey questions
- removal of variables with high missingness
- listwise deletion of remaining missing cases
- encoding of categorical variables
- creation of two dataset variants:
  - with satisfaction-related items
  - without satisfaction-related items

The dataset without satisfaction-related items was used as the main basis for the SHAP and DoWhy analyses. This was done to reduce conceptual overlap between predictors and the happiness outcome.

## Models

Three machine learning models were trained:

- Random Forest Regression
- Support Vector Regression
- XGBoost Regression

Each model was estimated in two configurations:

- with satisfaction-related items
- without satisfaction-related items

Model performance was evaluated using:

- RMSE
- R2

## SHAP Analysis

SHAP values were used to interpret the machine learning models.

The SHAP analysis identifies which variables contributed most strongly to the model predictions. The focus is on the most relevant predictors and broader predictor categories, rather than only on individual model rankings.

Important predictor categories include:

- health
- financial conditions
- meaning in life
- autonomy
- social relationships
- trust

## Causal Analysis with DoWhy

DoWhy was used to estimate possible causal effects for selected predictors.

Treatment variables were selected based on the SHAP results. For each treatment, a set of confounders was defined based on theory and previous research. Average Treatment Effects (ATEs) were estimated using backdoor adjustment with a linear regression estimator.

The DoWhy results are interpreted as additional evidence. They are not treated as proof of causal effects, because the analysis is based on observational survey data and depends on the selected confounders.

Refutation tests were also conducted:

- placebo treatment refuter
- random common cause refuter
- data subset refuter

These tests were used to examine whether the estimated effects remained stable under simple modifications of the analysis.

## Subgroup Analysis

Subgroup analyses were conducted for:

- gender
- age groups
- income quartiles

The subgroup analyses used the same treatment variables and confounder sets as the overall DoWhy analysis. This was done to keep the estimated effects comparable across groups.

## Repository Structure

```text
.
|-- Psycho_DataspellProjekt/          # Main analysis project
|   |-- data/                         # Raw, decoded, filtered, and cleaned EQLS data
|   |-- CausalAnalysis/               # DoWhy causal analysis notebooks
|   |   |-- Results/                  # ATE tables and refutation test results
|   |-- RandomForestRegression/       # Random Forest scripts, notebooks, and SHAP results
|   |   |-- Results/                  # Random Forest SHAP outputs
|   |-- SVR/                          # Support Vector Regression scripts, notebooks, and SHAP results
|   |   |-- Results/                  # SVR SHAP outputs
|   |-- XGBoost/                      # XGBoost scripts, notebooks, and SHAP results
|   |   |-- Results/                  # XGBoost SHAP outputs
|   |-- SubgroupAnalysis/             # Gender, age, income, and country subgroup analyses
|   |   |-- Results/                  # Subgroup-specific SHAP and DoWhy outputs
|   |-- pre_processing.ipynb          # Data preprocessing workflow
|   `-- question_filtering.py         # Script for filtering survey questions
|-- docs/                             # Additional EQLS data and concordance files
|-- backUpDaten/                      # Backup copies of EQLS data files
|-- Quellen/                          # Source material and references
`-- README.md
```
