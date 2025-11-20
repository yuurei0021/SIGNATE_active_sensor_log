"""
Data Preparation Script
Combine all individual sensor CSV files into single DataFrames and save as parquet files.
This script only needs to be run once.
"""

import pandas as pd
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR.parent / "raw"
TRAIN_DIR = RAW_DIR / "train"
TEST_DIR = RAW_DIR / "test"
TRAIN_MASTER = RAW_DIR / "train_master.csv"

OUTPUT_TRAIN = BASE_DIR / "train_combined.csv"
OUTPUT_TEST = BASE_DIR / "test_combined.csv"

def load_and_combine_train():
    """Load all training files and combine into a single DataFrame."""
    print("Loading training master data...")
    train_master = pd.read_csv(TRAIN_MASTER)

    print(f"Loading and combining {len(train_master)} training files...")
    train_list = []

    for idx, row in train_master.iterrows():
        file_id = row['id']
        file_path = TRAIN_DIR / f"{file_id}.csv"

        # Load sensor data
        df = pd.read_csv(file_path)

        # Add file_id column
        df['id'] = file_id

        # Add row index within each file (time step)
        df['time_step'] = range(len(df))

        train_list.append(df)

        # Progress indicator
        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{len(train_master)} files")

    # Combine all files
    train_combined = pd.concat(train_list, ignore_index=True)

    # Merge with class labels
    train_combined = train_combined.merge(train_master, on='id', how='left')

    print(f"Training data combined: {len(train_combined)} rows, {len(train_master)} files")
    return train_combined

def load_and_combine_test():
    """Load all test files and combine into a single DataFrame."""
    test_files = sorted(TEST_DIR.glob("test_*.csv"))

    print(f"Loading and combining {len(test_files)} test files...")
    test_list = []

    for idx, file_path in enumerate(test_files):
        file_id = file_path.stem

        # Load sensor data
        df = pd.read_csv(file_path)

        # Add file_id column
        df['id'] = file_id

        # Add row index within each file (time step)
        df['time_step'] = range(len(df))

        test_list.append(df)

        # Progress indicator
        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{len(test_files)} files")

    # Combine all files
    test_combined = pd.concat(test_list, ignore_index=True)

    print(f"Test data combined: {len(test_combined)} rows, {len(test_files)} files")
    return test_combined

def main():
    print("="*80)
    print("Data Preparation: Combining Individual Sensor Files")
    print("="*80)

    # Load and combine training data
    train_combined = load_and_combine_train()

    # Save as CSV
    print(f"\nSaving training data to {OUTPUT_TRAIN}...")
    train_combined.to_csv(OUTPUT_TRAIN, index=False)
    print(f"Training data saved: {OUTPUT_TRAIN}")
    print(f"  Shape: {train_combined.shape}")
    print(f"  Columns: {list(train_combined.columns)}")
    print(f"  Memory usage: {train_combined.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # Load and combine test data
    test_combined = load_and_combine_test()

    # Save as CSV
    print(f"\nSaving test data to {OUTPUT_TEST}...")
    test_combined.to_csv(OUTPUT_TEST, index=False)
    print(f"Test data saved: {OUTPUT_TEST}")
    print(f"  Shape: {test_combined.shape}")
    print(f"  Columns: {list(test_combined.columns)}")
    print(f"  Memory usage: {test_combined.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    print("\n" + "="*80)
    print("Data preparation complete!")
    print("="*80)
    print("\nYou can now use these combined files in your experiments:")
    print(f"  Train: {OUTPUT_TRAIN}")
    print(f"  Test: {OUTPUT_TEST}")

if __name__ == "__main__":
    main()
