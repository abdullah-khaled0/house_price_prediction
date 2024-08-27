import os
import sys
import pandas as pd


def get_df(file_path):
    """Read the input CSV file and return a DataFrame."""
    try:
        df = pd.read_csv(file_path)
        sys.stderr.write(f"The input DataFrame {file_path} size is {df.shape}\n")
        return df
    except Exception as e:
        sys.stderr.write(f"Error reading the file {file_path}: {e}\n")
        sys.exit(1)


def save_df(df, filename="prepared_data.csv"):
    """Save the DataFrame to the specified directory."""

    if not os.path.exists("data/prepare"):
        os.makedirs("data/prepare")

    output_dir = os.path.join("data", "prepare")

    output_path = os.path.join(output_dir, filename)
    try:
        df.to_csv(output_path, index=False)
        sys.stderr.write(f"DataFrame saved to {output_path}\n")
    except Exception as e:
        sys.stderr.write(f"Error saving the DataFrame: {e}\n")
        sys.exit(1)


def main():
    # if len(sys.argv) != 1:
    #     sys.stderr.write("Arguments error. Usage: python script.py <input_data_path>\n")
    #     sys.exit(1)

    in_path = sys.argv[1]
    data_input = os.path.join(in_path)

    dataset_df = get_df(data_input)
    dataset_df = dataset_df.drop('Id', axis=1)
    save_df(dataset_df)


if __name__ == "__main__":
    main()
