import os
import sys
import yaml

import numpy as np
import pandas as pd


from sklearn.preprocessing import LabelEncoder

def preprocess_data(dataset):
    
    # Calculate the percentage of null values in each column
    null_percentages = dataset.isnull().mean()

    # Drop columns with null percentage greater than 50%
    dataset_df = dataset.loc[:, null_percentages < 0.5]
    
    # Select object columns
    cat_cols = dataset.select_dtypes(include=['object']).columns

    # Encode the selected columns using LabelEncoder
    le = LabelEncoder()
    encoded_cols = dataset[cat_cols].apply(le.fit_transform)

    # Drop the original categorical columns
    dataset = dataset.drop(cat_cols, axis=1)

    # Add the encoded columns to the original DataFrame
    dataset = dataset.join(encoded_cols)
    
    dataset["LotFrontage"] = dataset["LotFrontage"].fillna(dataset["LotFrontage"].median())
    dataset["GarageYrBlt"] = dataset["GarageYrBlt"].fillna(dataset["GarageYrBlt"].median())
    dataset["MasVnrArea"] = dataset["MasVnrArea"].fillna(dataset["MasVnrArea"].median())
    
    return dataset


def get_df(file_path):
    """Read the input CSV file and return a DataFrame."""
    try:
        df = pd.read_csv(file_path)
        sys.stderr.write(f"The input DataFrame {file_path} size is {df.shape}\n")
        return df
    except Exception as e:
        sys.stderr.write(f"Error reading the file {file_path}: {e}\n")
        sys.exit(1)




def main():

    if len(sys.argv) != 1:
        sys.stderr.write("Arguments error.\n")
        sys.exit(1)

    params = yaml.safe_load(open("params.yaml"))["prepare"]

    split = params["split"]

    in_path = sys.argv[1]

    dataset_df = 

    dataset_df = preprocess_data(dataset_df)


if __name__ == "__main__":
    main()
