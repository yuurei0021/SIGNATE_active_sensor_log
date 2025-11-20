"""
Investigation of test_01014: Visual Analysis

Purpose: Analyze test_01014 sensor data to determine if it's more similar
         to walking or stairs patterns through visualization and feature comparison.

Background:
- Baseline model: 86.62% walking, 12.44% stairs
- Improved model: 62.13% walking, 37.24% stairs
- Ground truth: Unknown

Approach:
1. Load test_01014 sensor data
2. Select representative walking and stairs samples from train data
3. Visualize time series patterns
4. Compare statistical features
5. Make visual assessment
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (16, 10)


def load_sensor_data(file_id, data_type='train'):
    """Load sensor data from individual CSV file"""
    data_dir = project_root / 'data' / 'raw' / data_type
    file_path = data_dir / f'{file_id}.csv'

    df = pd.read_csv(file_path)
    df['time_step'] = range(len(df))
    df['id'] = file_id

    return df


def calculate_features(df):
    """Calculate key features from sensor data"""
    features = {}

    # Time domain features
    for axis in ['accelerometer_X', 'accelerometer_Y', 'accelerometer_Z']:
        features[f'{axis}_mean'] = df[axis].mean()
        features[f'{axis}_std'] = df[axis].std()
        features[f'{axis}_min'] = df[axis].min()
        features[f'{axis}_max'] = df[axis].max()
        features[f'{axis}_range'] = df[axis].max() - df[axis].min()

    # Magnitude
    df['magnitude'] = np.sqrt(
        df['accelerometer_X']**2 +
        df['accelerometer_Y']**2 +
        df['accelerometer_Z']**2
    )
    features['magnitude_mean'] = df['magnitude'].mean()
    features['magnitude_std'] = df['magnitude'].std()

    # Z-axis specific (stairs characteristic)
    features['Z_peaks'] = len(df[df['accelerometer_Z'] > df['accelerometer_Z'].mean() + df['accelerometer_Z'].std()])
    features['Z_valleys'] = len(df[df['accelerometer_Z'] < df['accelerometer_Z'].mean() - df['accelerometer_Z'].std()])

    return features


def plot_comparison(test_df, train_samples, output_dir):
    """Create comprehensive comparison plots"""

    # 1. Time series comparison (3x3 grid: test + 4 walking + 4 stairs)
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle('Sensor Data Comparison: test_01014 vs Representative Train Samples', fontsize=16, fontweight='bold')

    # Plot test_01014
    ax = axes[0, 0]
    ax.plot(test_df['time_step'], test_df['accelerometer_X'], label='X', alpha=0.7)
    ax.plot(test_df['time_step'], test_df['accelerometer_Y'], label='Y', alpha=0.7)
    ax.plot(test_df['time_step'], test_df['accelerometer_Z'], label='Z', alpha=0.7)
    ax.set_title('test_01014 (Unknown)', fontweight='bold', fontsize=12)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Acceleration (m/s²)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot walking samples
    for i, sample_id in enumerate(train_samples['walking'][:4]):
        row = (i // 2) + 1
        col = i % 2
        ax = axes[row, col] if row < 3 else None

        if ax and i < 4:
            df = load_sensor_data(sample_id)
            ax.plot(df['time_step'], df['accelerometer_X'], label='X', alpha=0.7)
            ax.plot(df['time_step'], df['accelerometer_Y'], label='Y', alpha=0.7)
            ax.plot(df['time_step'], df['accelerometer_Z'], label='Z', alpha=0.7)
            ax.set_title(f'WALKING: {sample_id}', fontweight='bold', fontsize=10, color='blue')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Acceleration (m/s²)')
            ax.legend()
            ax.grid(True, alpha=0.3)

    # Plot stairs samples
    start_idx = 2
    for i, sample_id in enumerate(train_samples['stairs'][:4]):
        idx = start_idx + i
        row = idx // 3
        col = idx % 3

        if row < 3 and col < 3:
            ax = axes[row, col]
            df = load_sensor_data(sample_id)
            ax.plot(df['time_step'], df['accelerometer_X'], label='X', alpha=0.7)
            ax.plot(df['time_step'], df['accelerometer_Y'], label='Y', alpha=0.7)
            ax.plot(df['time_step'], df['accelerometer_Z'], label='Z', alpha=0.7)
            ax.set_title(f'STAIRS: {sample_id}', fontweight='bold', fontsize=10, color='red')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Acceleration (m/s²)')
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_timeseries.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'comparison_timeseries.png'}")
    plt.close()

    # 2. Focus on Z-axis (stairs characteristic)
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle('Z-axis Focus: Stairs Climbing Pattern Analysis', fontsize=16, fontweight='bold')

    # Plot test_01014 Z-axis
    ax = axes[0, 0]
    ax.plot(test_df['time_step'], test_df['accelerometer_Z'], color='black', linewidth=2)
    ax.axhline(test_df['accelerometer_Z'].mean(), color='green', linestyle='--', label='Mean', alpha=0.7)
    ax.axhline(test_df['accelerometer_Z'].mean() + test_df['accelerometer_Z'].std(), color='red', linestyle='--', label='+1 STD', alpha=0.5)
    ax.axhline(test_df['accelerometer_Z'].mean() - test_df['accelerometer_Z'].std(), color='red', linestyle='--', label='-1 STD', alpha=0.5)
    ax.set_title('test_01014 Z-axis (Unknown)', fontweight='bold', fontsize=12)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Z Acceleration (m/s²)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot walking samples Z-axis
    for i, sample_id in enumerate(train_samples['walking'][:4]):
        row = (i // 2) + 1
        col = i % 2
        ax = axes[row, col] if row < 3 else None

        if ax and i < 4:
            df = load_sensor_data(sample_id)
            ax.plot(df['time_step'], df['accelerometer_Z'], color='blue', linewidth=2)
            ax.axhline(df['accelerometer_Z'].mean(), color='green', linestyle='--', alpha=0.7)
            ax.set_title(f'WALKING: {sample_id}', fontweight='bold', fontsize=10, color='blue')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Z Acceleration (m/s²)')
            ax.grid(True, alpha=0.3)

    # Plot stairs samples Z-axis
    start_idx = 2
    for i, sample_id in enumerate(train_samples['stairs'][:4]):
        idx = start_idx + i
        row = idx // 3
        col = idx % 3

        if row < 3 and col < 3:
            ax = axes[row, col]
            df = load_sensor_data(sample_id)
            ax.plot(df['time_step'], df['accelerometer_Z'], color='red', linewidth=2)
            ax.axhline(df['accelerometer_Z'].mean(), color='green', linestyle='--', alpha=0.7)
            ax.set_title(f'STAIRS: {sample_id}', fontweight='bold', fontsize=10, color='red')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Z Acceleration (m/s²)')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_z_axis.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'comparison_z_axis.png'}")
    plt.close()

    # 3. Feature comparison
    print("\n" + "="*80)
    print("FEATURE COMPARISON")
    print("="*80)

    # Calculate features for all samples
    test_features = calculate_features(test_df)

    walking_features = []
    for sample_id in train_samples['walking']:
        df = load_sensor_data(sample_id)
        walking_features.append(calculate_features(df))

    stairs_features = []
    for sample_id in train_samples['stairs']:
        df = load_sensor_data(sample_id)
        stairs_features.append(calculate_features(df))

    # Compare key features
    print(f"\n{'Feature':<30} {'test_01014':>15} {'Walking Avg':>15} {'Stairs Avg':>15}")
    print("-"*80)

    key_features = [
        'accelerometer_Y_std', 'accelerometer_Z_std',
        'magnitude_mean', 'magnitude_std',
        'Z_peaks', 'Z_valleys', 'accelerometer_Z_range'
    ]

    for feature in key_features:
        test_val = test_features[feature]
        walk_avg = np.mean([f[feature] for f in walking_features])
        stairs_avg = np.mean([f[feature] for f in stairs_features])

        print(f"{feature:<30} {test_val:>15.4f} {walk_avg:>15.4f} {stairs_avg:>15.4f}")

    # Calculate similarity scores (normalized distance)
    test_vec = np.array([test_features[f] for f in key_features])
    walk_vec = np.array([np.mean([feat[f] for feat in walking_features]) for f in key_features])
    stairs_vec = np.array([np.mean([feat[f] for feat in stairs_features]) for f in key_features])

    # Normalize
    all_vecs = np.vstack([test_vec, walk_vec, stairs_vec])
    mean = all_vecs.mean(axis=0)
    std = all_vecs.std(axis=0)
    std[std == 0] = 1  # Avoid division by zero

    test_norm = (test_vec - mean) / std
    walk_norm = (walk_vec - mean) / std
    stairs_norm = (stairs_vec - mean) / std

    dist_to_walk = np.linalg.norm(test_norm - walk_norm)
    dist_to_stairs = np.linalg.norm(test_norm - stairs_norm)

    print(f"\n{'Similarity Metric':<30} {'Value':>15}")
    print("-"*50)
    print(f"{'Distance to Walking':<30} {dist_to_walk:>15.4f}")
    print(f"{'Distance to Stairs':<30} {dist_to_stairs:>15.4f}")

    if dist_to_walk < dist_to_stairs:
        print(f"\n=> test_01014 is CLOSER to WALKING by {((dist_to_stairs - dist_to_walk) / dist_to_stairs * 100):.1f}%")
    else:
        print(f"\n=> test_01014 is CLOSER to STAIRS by {((dist_to_walk - dist_to_stairs) / dist_to_walk * 100):.1f}%")

    return test_features, walking_features, stairs_features


def main():
    print("="*80)
    print("test_01014 INVESTIGATION: Visual Analysis and Pattern Matching")
    print("="*80)

    output_dir = Path(__file__).parent

    # Load test_01014
    print("\nLoading test_01014 data...")
    test_df = load_sensor_data('test_01014', data_type='test')
    print(f"test_01014 shape: {test_df.shape}")

    # Load train master to select representative samples
    print("\nLoading train master...")
    train_master = pd.read_csv(project_root / 'data' / 'raw' / 'train_master.csv')

    # Select representative samples (first 8 of each class for consistency)
    walking_samples = train_master[train_master['class'] == 'walking']['id'].head(8).tolist()
    stairs_samples = train_master[train_master['class'] == 'stairs']['id'].head(8).tolist()

    train_samples = {
        'walking': walking_samples,
        'stairs': stairs_samples
    }

    print(f"Selected {len(walking_samples)} walking samples")
    print(f"Selected {len(stairs_samples)} stairs samples")

    # Create visualizations and compare features
    print("\nCreating visualizations...")
    test_features, walking_features, stairs_features = plot_comparison(test_df, train_samples, output_dir)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\nModel Predictions:")
    print("  Baseline:  86.62% walking, 12.44% stairs")
    print("  Improved:  62.13% walking, 37.24% stairs")
    print("\nVisual Analysis:")
    print("  See generated plots for pattern comparison")
    print("\nNext Steps:")
    print("  1. Examine comparison_timeseries.png")
    print("  2. Examine comparison_z_axis.png")
    print("  3. Compare feature statistics above")
    print("  4. Make informed decision based on visual evidence")


if __name__ == '__main__':
    main()
