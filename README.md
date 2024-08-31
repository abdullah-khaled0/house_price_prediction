# Project Title

A Comprehensive Machine Learning Pipeline for Data Preprocessing, Training, and Evaluation using DVC

## Overview

This repository contains a series of Python scripts designed to create a complete end-to-end machine learning pipeline. The pipeline covers the following key stages:

1. **Data Preprocessing**: Load, clean, and preprocess data.
2. **Model Training**: Train an XGBoost model using the preprocessed data.
3. **Model Evaluation**: Evaluate the trained model and log important metrics and plots.

## Requirements

The following Python packages are required to run the scripts:

- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `dvclive`
- `pyyaml`
- `dvc`


### Cloning the Repository

To clone this repository to your local machine, use the following command in your terminal:

```bash
git clone https://github.com/abdullah0150/house_pricing_prediction.git
```

You can install the required packages using `pip`:

```bash
pip install -r requirements.txt
```

## Role of DVC

DVC (Data Version Control) is used in this project to manage and version the entire machine learning pipeline. It provides the following functionalities:

- **Pipeline Management:** DVC defines and executes the steps in the pipeline (data preprocessing, model training, evaluation) using the `dvc.yaml` file.
- **Reproducibility:** DVC tracks changes in the data, code, and parameters, ensuring that the entire pipeline can be reproduced exactly, even if changes are made over time.
- **Data Versioning:** DVC handles large datasets by storing and versioning them outside the Git repository, allowing you to track and switch between different dataset versions easily.


## Data Preprocessing

The `preprocess.py` script handles various data preprocessing tasks, including data loading, cleaning, encoding categorical variables, and splitting the dataset into training and testing sets.

### Usage

To run the preprocessing script, use the following command:

```bash
python preprocess.py <input_data_directory>
```

## Model Training

The `train.py` script trains an XGBoost model using the preprocessed training data and hyperparameters specified in `params.yaml`.

### Usage

To run the training script, use the following command:

```bash
python train.py <input_data_directory> <output_model_path>
```


## Model Evaluation

The `evaluate.py` script evaluates the trained XGBoost model using the test data.

### Usage

To run the evaluation script, use the following command:

```bash
python evaluate.py <model_file> <input_data_directory>
```

## Reproducing the Pipeline with DVC

To fully reproduce the pipeline using DVC, follow these steps:

### 1. Initialize DVC

If you haven't already, initialize DVC in your project directory:

```bash
dvc init
```

### 2. Run the Pipeline

Use DVC to execute the entire pipeline, which will automatically run all the steps defined in `dvc.yaml`:

```bash
dvc repro
```

### 3. Track Changes and Version Control

DVC allows you to track and version control your data and models. After running the pipeline, use the following commands to save changes:

```bash
dvc add data/processed
dvc add model/xgb_model.pkl
git add data.dvc model.dvc dvc.lock
git commit -m "Add processed data and trained model"
```

### 4. Share and Collaborate

You can push your DVC-managed data and models to remote storage (e.g., S3, Google Drive) for collaboration:

```bash
dvc remote add -d myremote <REMOTE_STORAGE_URL>
dvc push
```
