import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = REPO_ROOT / "data" / "raw"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

def validate_sensor_file(file_path):
    """センサーデータファイルの検証"""
    issues = []

    try:
        df = pd.read_csv(file_path)

        # カラム名チェック
        expected_cols = ['accelerometer_X', 'accelerometer_Y', 'accelerometer_Z']
        if list(df.columns) != expected_cols:
            issues.append(f"Unexpected columns: {list(df.columns)}")

        # 行数チェック
        if len(df) != 30:
            issues.append(f"Expected 30 rows, got {len(df)}")

        # 欠損値チェック
        if df.isnull().any().any():
            issues.append(f"Contains NaN values: {df.isnull().sum().to_dict()}")

        # 数値型チェック
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                issues.append(f"Column {col} is not numeric")

        # 無限大チェック
        if np.isinf(df.values).any():
            issues.append("Contains infinite values")

        return df, issues

    except Exception as e:
        return None, [f"Failed to read file: {str(e)}"]

def validate_dataset(data_dir, dataset_name):
    """データセット全体の検証"""
    print(f"\n{'='*60}")
    print(f"Validating {dataset_name} dataset")
    print(f"{'='*60}")

    files = sorted(data_dir.glob("*.csv"))
    total_files = len(files)

    print(f"Total files: {total_files}")

    valid_count = 0
    invalid_files = []
    issue_summary = defaultdict(int)

    for i, file_path in enumerate(files):
        df, issues = validate_sensor_file(file_path)

        if issues:
            invalid_files.append((file_path.name, issues))
            for issue in issues:
                issue_summary[issue] += 1
        else:
            valid_count += 1

        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1}/{total_files} files...")

    print(f"\nResults:")
    print(f"  Valid files: {valid_count}/{total_files}")
    print(f"  Invalid files: {len(invalid_files)}/{total_files}")

    if issue_summary:
        print(f"\nIssue summary:")
        for issue, count in sorted(issue_summary.items(), key=lambda x: -x[1]):
            print(f"  {issue}: {count} files")

    if invalid_files and len(invalid_files) <= 10:
        print(f"\nInvalid files:")
        for filename, issues in invalid_files:
            print(f"  {filename}: {issues}")

    return valid_count == total_files, invalid_files

def validate_train_master():
    """train_master.csvの検証"""
    print(f"\n{'='*60}")
    print("Validating train_master.csv")
    print(f"{'='*60}")

    master_file = DATA_DIR / "train_master.csv"
    df = pd.read_csv(master_file)

    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nClass distribution:")
    print(df['class'].value_counts())

    # ID検証
    expected_ids = {f"train_{i:05d}" for i in range(len(df))}
    actual_ids = set(df['id'])

    missing_ids = expected_ids - actual_ids
    extra_ids = actual_ids - expected_ids

    if missing_ids:
        print(f"\nMissing IDs: {sorted(list(missing_ids))[:10]}...")
    if extra_ids:
        print(f"\nExtra IDs: {sorted(list(extra_ids))[:10]}...")

    # 欠損値チェック
    if df.isnull().any().any():
        print(f"\nNaN values found:")
        print(df.isnull().sum())
    else:
        print("\nNo NaN values found")

    return df

def main():
    print("Data Validation Report")
    print("="*60)

    # train_master.csvの検証
    train_master = validate_train_master()

    # trainデータの検証
    train_valid, train_invalid = validate_dataset(TRAIN_DIR, "train")

    # testデータの検証
    test_valid, test_invalid = validate_dataset(TEST_DIR, "test")

    # サマリー
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"train_master.csv: OK")
    print(f"train dataset: {'OK' if train_valid else 'FAILED'}")
    print(f"test dataset: {'OK' if test_valid else 'FAILED'}")

    if train_valid and test_valid:
        print("\n[OK] All data is ready for modeling (no preprocessing needed)")
    else:
        print("\n[FAILED] Data issues found - preprocessing may be required")

if __name__ == "__main__":
    main()
