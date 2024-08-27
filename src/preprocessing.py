import os
import sys
import yaml

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split



def preprocess_data(dataset: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the dataset by handling missing values and encoding categorical features."""
    # Drop columns with more than 50% missing values
    null_percentages = dataset.isnull().mean()
    dataset = dataset.loc[:, null_percentages < 0.5]
    
    # Encode categorical columns
    cat_cols = dataset.select_dtypes(include=['object']).columns
    if not cat_cols.empty:
        le = LabelEncoder()
        dataset[cat_cols] = dataset[cat_cols].apply(le.fit_transform)

    # Fill missing values with the median for specified columns
    cols_to_fill = ["LotFrontage", "GarageYrBlt", "MasVnrArea"]
    for col in cols_to_fill:
        if col in dataset.columns:
            dataset[col] = dataset[col].fillna(dataset[col].median())

    return dataset


def load_dataframe(file_path: str) -> pd.DataFrame:
    """Read the input CSV file and return a DataFrame."""
    try:
        df = pd.read_csv(file_path)
        sys.stderr.write(f"The input DataFrame {file_path} size is {df.shape}\n")
        return df
    except Exception as e:
        sys.stderr.write(f"Error reading the file {file_path}: {e}\n")
        sys.exit(1)


def split_data(dataset_df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42):
    """Split the dataset into training and testing sets."""
    X = dataset_df.drop(target_col, axis=1)
    y = dataset_df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def scale_data(X_train: pd.DataFrame, X_test: pd.DataFrame) -> (np.ndarray, np.ndarray): # type: ignore
    """Scale the features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def save_processed_data(X_train, X_test, y_train, y_test, output_dir: str):
    """Save the processed data to the specified output directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    pd.DataFrame(X_train).to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    pd.DataFrame(X_test).to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    pd.DataFrame(y_train).to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    pd.DataFrame(y_test).to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    sys.stderr.write(f"Processed data saved to {output_dir}\n")


def main():
    # if len(sys.argv) != 2:
    #     sys.stderr.write("Usage: python script.py <input_data_directory>\n")
    #     sys.exit(1)

    input_data_dir = sys.argv[1]
    data_input = os.path.join(input_data_dir)

    # Load parameters from YAML file
    try:
        with open("params.yaml", "r") as file:
            params = yaml.safe_load(file)["processing"]
        split = params["split"]
    except Exception as e:
        sys.stderr.write(f"Error reading params.yaml: {e}\n")
        sys.exit(1)

    # Load and preprocess the data
    dataset_df = load_dataframe(data_input)
    dataset_df = preprocess_data(dataset_df)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(dataset_df, target_col='SalePrice', test_size=split)

    # Scale the features
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    # Save the processed data
    output_dir = os.path.join("data", "processed")
    save_processed_data(X_train_scaled, X_test_scaled, y_train, y_test, output_dir)


if __name__ == "__main__":
    main()
