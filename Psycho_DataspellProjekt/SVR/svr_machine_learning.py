from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import pandas as pd
import shap
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV

# Pipeline
def prepare_data(df, target_column, test_size=0.2, random_state=42):
    """
    Prepares the data for training by splitting it into training and test data.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_svr_model(X_train, y_train, kernel='rbf', C=1.0, epsilon=0.1):
    """
    Trains the model using Support Vector Regression (SVR).
    Scaling is required for SVR.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = SVR(kernel=kernel, C=C, epsilon=epsilon)
    model.fit(X_train_scaled, y_train)

    return model, scaler

def evaluate_model(model, scaler, X_test, y_test):
    """
    Evaluates the model by calculating the mean squared error and r2 score.
    """
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f" RMSE: {rmse:.4f}")
    print(f" R²: {r2:.4f}")

    return rmse, r2

def calculate_shap_svr(model, scaler, X_sample, number_of_rows, number_of_head):
    """
    Calculates the Shapley values of the model using KernelExplainer (for non-tree models like SVR).
    """
    print(" Starting SHAP for " + str(number_of_rows) + " rows...")

    # Scale sample
    X_scaled = scaler.transform(X_sample)

    # KernelExplainer needs a small background dataset
    explainer = shap.KernelExplainer(model.predict, X_scaled[:50])

    # Calculate SHAP values
    shap_values = explainer.shap_values(X_scaled, nsamples=100)

    # Save SHAP values as dataframe
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


def run_svr_pipeline(df, target_column, test_size=0.2, random_state=42,
                     shap_rows=1000, number_header_rows=30):
    """
    Runs the SVR baseline model plus the calculation of SHAP values for it.
    """
    X_train, X_test, y_train, y_test = prepare_data(df, target_column, test_size, random_state)

    model, scaler = train_svr_model(X_train, y_train)
    evaluate_model(model, scaler, X_test, y_test)

    # SHAP only for a sample of the data
    X_sample = X_test.iloc[:shap_rows]
    shap_results = calculate_shap_svr(model, scaler, X_sample, shap_rows, number_header_rows)

    return model, shap_results


def run_svr_gridsearch_and_shap(df, target_column, test_size=0.2, random_state=42,
                                cv_folds=3, shap_rows=1000, number_header_rows=30):
    """
    Runs a compact GridSearchCV on Support Vector Regression (SVR).
    Includes scaling and SHAP values.
    """
    X_train, X_test, y_train, y_test = prepare_data(df, target_column, test_size, random_state)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    param_grid = {
        'C': [0.1, 1, 10],
        'epsilon': [0.01, 0.1, 0.2],
        'kernel': ['rbf']
    }

    svr = SVR()

    grid_search = GridSearchCV(
        estimator=svr,
        param_grid=param_grid,
        scoring='neg_root_mean_squared_error',
        cv=cv_folds,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train_scaled, y_train)

    best_model = grid_search.best_estimator_

    # Evaluate
    evaluate_model(best_model, scaler, X_test, y_test)

    # SHAP only for a sample of the data
    X_sample = X_test.iloc[:shap_rows]
    shap_results = calculate_shap_svr(best_model, scaler, X_sample, shap_rows, number_header_rows)

    print(f"\n Best Parameters: {grid_search.best_params_}")

    return best_model, shap_results
