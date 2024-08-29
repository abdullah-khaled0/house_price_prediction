import os
import pickle
import sys

import pandas as pd
from sklearn import metrics

from dvclive import Live
from matplotlib import pyplot as plt


def evaluate(model, X, y, live):
    """
    Evaluate the model using the provided data and log metrics and plots.

    Args:
        model (XGBRegressor): Trained XGBoost model.
        X (pd.DataFrame): Features to evaluate.
        y (pd.Series): True labels.
        live (dvclive.Live): DVCLive instance for logging.
    """
    # Make predictions
    predictions = model.predict(X)

    # Calculate regression metrics
    mse = metrics.mean_squared_error(y, predictions)
    rmse = mse ** 0.5
    mae = metrics.mean_absolute_error(y, predictions)
    r2 = metrics.r2_score(y, predictions)
    
    # Log metrics using DVCLive
    if not live.summary:
        live.summary = {"mse": {}, "rmse": {}, "mae": {}, "r2": {}}
    live.summary["mse"] = mse
    live.summary["rmse"] = rmse
    live.summary["mae"] = mae
    live.summary["r2"] = r2

    # Log the predicted vs actual values plot
    plt.figure(dpi=100)
    plt.scatter(y, predictions, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], '--r')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    live.log_image(f"actual_vs_predicted.png", plt.gcf())
    plt.close()

    # Log model artifact
    live.log_artifact("model/xgb_model.pkl", type="model")


def save_importance_plot(live, model, feature_names):
    """
    Save feature importance plot.

    Args:
        live (dvclive.Live): DVCLive instance.
        model (XGBRegressor): Trained XGBoost model.
        feature_names (list): List of feature names.
    """
    fig, axes = plt.subplots(dpi=100)
    fig.subplots_adjust(bottom=0.2, top=0.95)
    axes.set_ylabel("Feature Importance (Gain)")

    importances = model.feature_importances_
    xgb_importances = pd.Series(importances, index=feature_names).nlargest(n=30)
    xgb_importances.plot.bar(ax=axes)

    live.log_image("importance.png", fig)
    plt.close(fig)


def main():
    EVAL_PATH = "eval"

    # if len(sys.argv) != 3:
    #     sys.stderr.write("Arguments error. Usage:\n")
    #     sys.stderr.write("\tpython evaluate.py <model_file> <input_data_directory>\n")
    #     sys.exit(1)

    model_file = sys.argv[1]
    input_data_dir = sys.argv[2]

 
    test_file = os.path.join(input_data_dir, "X_test.csv")
    y_test_file = os.path.join(input_data_dir, "y_test.csv")

    # Load model and data
    with open(model_file, "rb") as fd:
        model = pickle.load(fd)

    X_test = pd.read_csv(test_file)
    y_test = pd.read_csv(y_test_file).squeeze()

    with open('data/column_names.txt', 'r') as file:
        feature_names = [line.strip() for line in file]
    

    # Evaluate train and test datasets
    with Live(EVAL_PATH, dvcyaml=False) as live:
        evaluate(model, X_test, y_test, live)

        # Dump feature importance plot
        save_importance_plot(live, model, feature_names)


if __name__ == "__main__":
    main()
