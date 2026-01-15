from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
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

def train_model(X_train, y_train, n_estimators=100, random_state=42):
    """
    Trains the model using the random forest regressor model.
    """
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model by calculating the mean squared error and r2 score.
    """
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"✅ RMSE: {rmse:.4f}")
    print(f"✅ R²: {r2:.4f}")
    return rmse, r2

def calculate_shap(model, X_sample, number_of_rows, number_of_head):
    """
    Calculates the Shapley values of the model.
    """
    # SHAP-Explainer
    explainer = shap.TreeExplainer(model)

    # Calculate Shaps with loading animation
    print("🔍 Starting SHAP  for " + str(number_of_rows) + " rows...")
    shap_values_list = []

    for i in tqdm(range(len(X_sample))):
        shap_row = explainer.shap_values(X_sample.iloc[[i]])
        shap_values_list.append(shap_row)

    shap_values = np.vstack(shap_values_list)
    #shap.summary_plot(shap_values, X_sample, max_display=20)

    # Save SHAP values as dataframe
    shap_df = pd.DataFrame(shap_values, columns=X_sample.columns)
    mean_abs_shap = shap_df.abs().mean()
    mean_signed_shap = shap_df.mean()

    # Combine into one DataFrame
    summary_df = pd.DataFrame({
        "Questions": mean_abs_shap.index,
        "Value": mean_abs_shap.values,
        "Direction": mean_signed_shap.values
    }).sort_values(by="Value", ascending=False)


    print("\nTop SHAP features:")
    print(summary_df.head(number_of_head))
    return summary_df


def run_pipeline(df, target_column, test_size=0.2, random_state=42, n_estimators=100, shap_rows=1000, number_header_rows=30):
    """
    Combines the above function to create a pipeline in order to run the Random Forest Regression model plus the calculation of SHAPly values of it
    """
    X_train, X_test, y_train, y_test = prepare_data(df, target_column, test_size, random_state)
    model = train_model(X_train, y_train, n_estimators, random_state)
    evaluate_model(model, X_test, y_test)

    # SHAP only for a sample of the data
    X_sample = X_test.iloc[:shap_rows]
    shap_results = calculate_shap(model, X_sample, shap_rows, number_header_rows)
    return model, shap_results


def run_rf_gridsearch_and_shap(df, target_column, test_size=0.2, random_state=42, cv_folds=3, shap_rows=1000, number_header_rows=30, ):
    """
    Runs a compact GridSearchCV on RandomForestRegressor.
    """
    X_train, X_test, y_train, y_test = prepare_data(df, target_column, test_size, random_state)
    # Define a parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, None],
        'max_features': ['sqrt', 'log2']
    }
    # Setup model and GridSearchCV
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rf,
                               param_grid=param_grid,
                               scoring='neg_root_mean_squared_error',
                               cv=cv_folds,
                               n_jobs=-1,
                               verbose=1)
    # Run the grid search
    grid_search.fit(X_train, y_train)
    # Evaluate
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    print(f"\n✅ Best Parameters: {grid_search.best_params_}")
    print(f"✅ Test RMSE: {rmse:.4f}")
    print(f"✅ Test R²: {r2:.4f}")

    # SHAP only for a sample of the data
    X_sample = X_test.iloc[:shap_rows]
    shap_results = calculate_shap(best_model, X_sample, shap_rows, number_header_rows)

    return best_model, shap_results
