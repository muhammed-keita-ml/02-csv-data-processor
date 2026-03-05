import pandas as pd
import numpy as np
import os
from datetime import datetime

# ─────────────────────────────────────────
# CSV Data Processor & Analysis Utility
# Reads, cleans, transforms, and exports
# ─────────────────────────────────────────

from typing import Final

INPUT_FILE: Final = "sample_data.csv"
OUTPUT_DIR: Final = "output"


def load_data(filepath):
    """Load CSV file into a DataFrame"""
    print(f"[LOAD] Reading {filepath}...")
    df = pd.read_csv(filepath)
    print(f"[LOAD] {len(df)} rows, {len(df.columns)} columns loaded.")
    return df


def explore_data(df):
    """Print basic info about the dataset"""
    print("\n[INFO] Dataset Overview")
    print("-" * 40)
    print(df.info())
    print("\n[INFO] First 5 rows:")
    print(df.head())
    print("\n[INFO] Missing values per column:")
    print(df.isnull().sum())
    print("\n[INFO] Basic statistics:")
    print(df.describe())


def clean_data(df):
    """Clean the dataset: remove duplicates, fill nulls"""
    original_len = len(df)

    # Remove duplicate rows
    df = df.drop_duplicates()
    print(f"[CLEAN] Removed {original_len - len(df)} duplicate rows.")

    # Fill missing numeric values with column mean
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

    # Fill missing text values with 'Unknown'
    str_cols = df.select_dtypes(include=["object"]).columns
    df[str_cols] = df[str_cols].fillna("Unknown")

    print(f"[CLEAN] Remaining rows: {len(df)}")
    print(f"[CLEAN] Missing values after cleaning: {df.isnull().sum().sum()}")
    return df


def transform_data(df):
    """Add useful derived columns"""
    # Add a processed timestamp column
    df["processed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # If there are numeric columns, add a row sum column
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        df["row_total"] = df[num_cols].sum(axis=1)
        print(f"[TRANSFORM] Added row_total from columns: {num_cols}")

    print("[TRANSFORM] Transformation complete.")
    return df


def export_data(df, output_dir):
    """Export cleaned data to CSV and summary to text"""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Export cleaned CSV
    csv_path = os.path.join(output_dir, f"cleaned_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    print(f"[EXPORT] Cleaned CSV saved: {csv_path}")

    # Export summary report
    summary_path = os.path.join(output_dir, f"summary_{timestamp}.txt")
    with open(summary_path, "w") as f:
        f.write("=== DATA PROCESSING SUMMARY ===\n")
        f.write(f"Processed at: {timestamp}\n")
        f.write(f"Total rows:   {len(df)}\n")
        f.write(f"Total cols:   {len(df.columns)}\n\n")
        f.write("Columns:\n")
        for col in df.columns:
            f.write(f"  - {col}\n")
    print(f"[EXPORT] Summary saved: {summary_path}")


if __name__ == "__main__":
    df = load_data(INPUT_FILE)
    explore_data(df)
    df = clean_data(df)
    df = transform_data(df)
    export_data(df, OUTPUT_DIR)
    print("\n[DONE] Processing complete!")
