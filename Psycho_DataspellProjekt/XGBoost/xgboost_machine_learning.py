from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import pandas as pd
import shap
import numpy as np
from tqdm import tqdm

# Pipeline
def prepare_data(df, target_column, test_size=0.2, random_state=42):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_xgb_model(X_train, y_train,
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42):
    """
    Trains a standard XGBoost regression model.
    Parameters can be modified for experiments.
    """
    model = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        objective="reg:squarederror",
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"✅ RMSE: {rmse:.4f}")
    print(f"✅ R²: {r2:.4f}")
    return rmse, r2

def calculate_shap(model, X_sample, number_of_rows, number_of_head):
    """
    Calculates SHAP values using TreeExplainer (works perfectly for XGBoost).
    """
    explainer = shap.TreeExplainer(model)

    print("🔍 Starting SHAP for " + str(number_of_rows) + " rows...")
    shap_values_list = []
    for i in tqdm(range(len(X_sample))):
        shap_row = explainer.shap_values(X_sample.iloc[[i]])
        shap_values_list.append(shap_row)

    shap_values = np.vstack(shap_values_list)

    shap_df = pd.DataFrame(shap_values, columns=X_sample.columns)
    mean_abs_shap = shap_df.abs().mean()
    mean_signed_shap = shap_df.mean()

    summary_df = pd.DataFrame({
        "Questions": mean_abs_shap.index,
        "Value": mean_abs_shap.values,
        "Direction": mean_signed_shap.values
    }).sort_values(by="Value", ascending=False)

    print("\nTop SHAP features:")
    print(summary_df.head(number_of_head))
    return summary_df


def run_xgb_pipeline(df, target_column, test_size=0.2, random_state=42,
                     shap_rows=1000, number_header_rows=30):
    """
    Baseline XGBoost + SHAP
    """
    X_train, X_test, y_train, y_test = prepare_data(df, target_column, test_size, random_state)
    model = train_xgb_model(X_train, y_train, random_state)
    evaluate_model(model, X_test, y_test)

    X_sample = X_test.iloc[:shap_rows]
    shap_results = calculate_shap(model, X_sample, shap_rows, number_header_rows)
    return model, shap_results


def run_xgb_gridsearch_and_shap(df, target_column, test_size=0.2,
                                random_state=42, cv_folds=3,
                                shap_rows=1000, number_header_rows=30):
    """
    GridSearchCV for XGBoost Regression (compact).
    """
    X_train, X_test, y_train, y_test = prepare_data(df, target_column, test_size, random_state)

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1, 0.2],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    }

    xgb = XGBRegressor(
        random_state=random_state,
        objective="reg:squarederror",
        n_jobs=-1
    )

    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring='neg_root_mean_squared_error',
        cv=cv_folds,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n✅ Best Parameters: {grid_search.best_params_}")
    print(f"✅ Test RMSE: {rmse:.4f}")
    print(f"✅ Test R²: {r2:.4f}")

    X_sample = X_test.iloc[:shap_rows]
    shap_results = calculate_shap(best_model, X_sample, shap_rows, number_header_rows)

    return best_model, shap_results
