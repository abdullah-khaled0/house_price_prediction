import os
import pickle
import sys
import yaml
import pandas as pd
from xgboost import XGBRegressor


def train(seed, n_estimators, max_depth, learning_rate, X_train, y_train):
    """
    Train an XGBoost model with the given parameters.
    
    Parameters:
        seed (int): Random seed for reproducibility.
        n_estimators (int): Number of boosting rounds.
        max_depth (int): Maximum tree depth for base learners.
        learning_rate (float): Learning rate (shrinkage).
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
    
    Returns:
        XGBRegressor: Trained XGBoost model.
    """
    # Create and train the XGBRegressor model
    xgb = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, 
                       learning_rate=learning_rate, random_state=seed)
    xgb.fit(X_train, y_train)

    return xgb


def main():
    # Load parameters from params.yaml
    params = yaml.safe_load(open("params.yaml"))["train"]

    # if len(sys.argv) != 3:
    #     sys.stderr.write("Arguments error. Usage:\n")
    #     sys.stderr.write("\tpython train.py <input_data_directory> <output_model_path>\n")
    #     sys.exit(1)

    input_data_dir = sys.argv[1]
    output_model_path = sys.argv[2]

    # Load processed training data
    X_train = pd.read_csv(os.path.join(input_data_dir, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(input_data_dir, "y_train.csv")).squeeze()  # .squeeze() to convert to Series

    # Retrieve hyperparameters from the params file
    seed = params["seed"]
    n_estimators = params["n_estimators"]
    max_depth = params["max_depth"]
    learning_rate = params["learning_rate"]

    # Train the model
    xgb = train(seed=seed, n_estimators=n_estimators, max_depth=max_depth, 
                learning_rate=learning_rate, X_train=X_train, y_train=y_train)

    # Save the trained model to the specified output path
    with open(output_model_path, "wb") as fd:
        pickle.dump(xgb, fd)
        sys.stderr.write(f"Model saved to {output_model_path}\n")


if __name__ == "__main__":
    main()
