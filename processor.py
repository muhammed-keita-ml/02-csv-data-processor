import pandas as pd
import numpy as np
import os
import mlflow
import dagshub
from datetime import datetime

# ── MLflow + DagsHub tracking setup ──────────────────
dagshub.init(
    repo_owner="muhammed-keita-ml", repo_name="02-csv-data-processor", mlflow=True
)

INPUT_FILE = "sample_data.csv"
OUTPUT_DIR = "output"


def load_data(filepath):
    """Load CSV file into a DataFrame"""
    df = pd.read_csv(filepath)
    print(f"[LOAD]    Loaded {len(df)} rows from {filepath}")
    return df


def explore_data(df):
    """Print dataset overview"""
    print(f"[EXPLORE] Shape: {df.shape}")
    print(f"[EXPLORE] Columns: {list(df.columns)}")
    print(f"[EXPLORE] Missing values:\n{df.isnull().sum()}")


def clean_data(df):
    """Remove duplicates and fill missing values"""
    original_rows = len(df)
    df = df.drop_duplicates()
    removed = original_rows - len(df)
    print(f"[CLEAN]   Removed {removed} duplicate rows")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    text_cols = df.select_dtypes(include=["object"]).columns
    df[text_cols] = df[text_cols].fillna("Unknown")

    remaining = int(df.isnull().sum().sum())
    print(f"[CLEAN]   Missing values remaining: {remaining}")
    return df


def transform_data(df):
    """Add derived columns"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df["row_total"] = df[numeric_cols].sum(axis=1)
    df["processed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[TRANSFORM] Added row_total and processed_at columns")
    return df


def export_data(df):
    """Export cleaned data to output folder"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(OUTPUT_DIR, f"cleaned_data_{timestamp}.csv")
    df.to_csv(output_path, index=False)
    print(f"[EXPORT]  Saved to {output_path}")

    summary_path = os.path.join(OUTPUT_DIR, f"summary_{timestamp}.txt")
    with open(summary_path, "w") as f:
        f.write(f"Rows: {len(df)}\n")
        f.write(f"Columns: {len(df.columns)}\n")
        f.write(f"Processed at: {timestamp}\n")
    print(f"[EXPORT]  Summary saved to {summary_path}")


if __name__ == "__main__":
    with mlflow.start_run():

        # Log input parameters
        mlflow.log_param("input_file", INPUT_FILE)

        # Load
        df = load_data(INPUT_FILE)
        mlflow.log_metric("original_rows", len(df))
        mlflow.log_metric("original_columns", len(df.columns))

        # Explore
        explore_data(df)

        # Clean
        df = clean_data(df)
        mlflow.log_metric("rows_after_cleaning", len(df))
        mlflow.log_metric("missing_values_remaining", int(df.isnull().sum().sum()))

        # Transform
        df = transform_data(df)

        # Export
        export_data(df)
        mlflow.log_metric("final_rows", len(df))
        mlflow.log_metric("final_columns", len(df.columns))

        print("[MLFLOW] Run tracked successfully on DagsHub.")
