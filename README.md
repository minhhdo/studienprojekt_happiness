# Master Thesis: Predicting and Explaining Happiness with Machine Learning and Causal Inference

This repository contains the code and analysis for my master thesis on the prediction and explanation of happiness using data from the European Quality of Life Survey (EQLS) 2011.

The project combines machine learning, SHAP-based model interpretation, and causal inference with DoWhy. The main goal is to identify relevant predictors of happiness and compare them with findings from psychological research on subjective well-being.

## Project Overview

The analysis focuses on the survey item:

> How happy are you?

This variable is used as the target variable. The predictors include health, financial conditions, social relationships, autonomy, meaning in life, trust, and other variables from the EQLS dataset.

The project follows three main steps:

1. Prediction of happiness using machine learning models
2. Interpretation of model predictions using SHAP values
3. Estimation of possible causal effects using DoWhy

## Data

The analysis is based on the European Quality of Life Survey 2011.

The raw data are not included in this repository because of data access restrictions. The dataset can be obtained from the official data provider.

Main preprocessing steps include:

- selection of relevant variables
- removal of variables with high missingness
- listwise deletion of remaining missing cases
- encoding of categorical variables
- creation of two dataset variants:
  - with satisfaction-related items
  - without satisfaction-related items

The second variant was used to reduce conceptual overlap between predictors and the happiness outcome.

## Models

Three machine learning models were trained and compared:

- Random Forest Regression
- Support Vector Regression
- XGBoost Regression

Each model was estimated in two configurations:

- with satisfaction-related items
- without satisfaction-related items

Model performance was evaluated using:

- RMSE
- R²

## SHAP Analysis

SHAP values were used to interpret the machine learning models.

The SHAP analysis identifies which variables contributed most strongly to the model predictions. The main focus is on stable predictor categories across models rather than on individual model-specific rankings.

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

