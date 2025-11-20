"""
Detailed analysis of problematic stairs samples
Compare sensor data patterns between correct and misclassified stairs
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent.parent / "data" / "processed"
RAW_DIR = BASE_DIR.parent.parent / "data" / "raw"

# Load data
print("="*80)
print("Detailed Stairs Sample Analysis")
print("="*80)

print("\nLoading combined training data...")
train_combined = pd.read_csv(DATA_DIR / "train_combined.csv")
print(f"Loaded: {len(train_combined)} rows")

print("\nLoading problematic samples...")
problematic = pd.read_csv(BASE_DIR / "stairs_problematic_samples.csv")
print(f"Problematic samples: {len(problematic)}")

# Get misclassified IDs
misclassified_ids = problematic[problematic['predicted_class'] != 'stairs']['id'].tolist()
print(f"\nMisclassified stairs IDs: {misclassified_ids}")

# Get low confidence but correct IDs
low_conf_correct_ids = problematic[
    (problematic['predicted_class'] == 'stairs') &
    (problematic['prob_stairs'] < 0.95)
]['id'].tolist()
print(f"Low confidence (but correct) IDs: {low_conf_correct_ids}")

# Get all stairs samples
all_stairs_ids = train_combined[train_combined['class'] == 'stairs']['id'].unique()
print(f"\nTotal stairs samples: {len(all_stairs_ids)}")

# High confidence stairs (for comparison)
high_conf_stairs_ids = [sid for sid in all_stairs_ids
                        if sid not in misclassified_ids + low_conf_correct_ids]
print(f"High confidence stairs samples: {len(high_conf_stairs_ids)}")

# ============================================================================
# Compare sensor data patterns
# ============================================================================
print("\n" + "="*80)
print("Sensor Data Pattern Comparison")
print("="*80)

def calculate_features(sensor_df):
    """Calculate features from sensor data"""
    features = {}

    for axis in ['accelerometer_X', 'accelerometer_Y', 'accelerometer_Z']:
        values = sensor_df[axis].values
        features[f'{axis}_mean'] = np.mean(values)
        features[f'{axis}_std'] = np.std(values)
        features[f'{axis}_min'] = np.min(values)
        features[f'{axis}_max'] = np.max(values)
        features[f'{axis}_range'] = np.max(values) - np.min(values)

    # Magnitude
    X = sensor_df['accelerometer_X'].values
    Y = sensor_df['accelerometer_Y'].values
    Z = sensor_df['accelerometer_Z'].values
    magnitude = np.sqrt(X**2 + Y**2 + Z**2)
    features['magnitude_mean'] = np.mean(magnitude)
    features['magnitude_std'] = np.std(magnitude)

    return features

# Calculate features for each group
print("\nCalculating features for misclassified stairs...")
misclassified_features = []
for file_id in misclassified_ids:
    sensor_df = train_combined[train_combined['id'] == file_id][
        ['accelerometer_X', 'accelerometer_Y', 'accelerometer_Z']
    ]
    features = calculate_features(sensor_df)
    features['id'] = file_id
    features['group'] = 'misclassified'
    misclassified_features.append(features)

print("Calculating features for low confidence stairs...")
low_conf_features = []
for file_id in low_conf_correct_ids:
    sensor_df = train_combined[train_combined['id'] == file_id][
        ['accelerometer_X', 'accelerometer_Y', 'accelerometer_Z']
    ]
    features = calculate_features(sensor_df)
    features['id'] = file_id
    features['group'] = 'low_confidence'
    low_conf_features.append(features)

print("Calculating features for high confidence stairs (sample)...")
sample_high_conf_ids = np.random.choice(high_conf_stairs_ids, size=min(20, len(high_conf_stairs_ids)), replace=False)
high_conf_features = []
for file_id in sample_high_conf_ids:
    sensor_df = train_combined[train_combined['id'] == file_id][
        ['accelerometer_X', 'accelerometer_Y', 'accelerometer_Z']
    ]
    features = calculate_features(sensor_df)
    features['id'] = file_id
    features['group'] = 'high_confidence'
    high_conf_features.append(features)

# Combine features
all_features = pd.DataFrame(misclassified_features + low_conf_features + high_conf_features)

# ============================================================================
# Statistical comparison
# ============================================================================
print("\n" + "="*80)
print("Feature Statistics by Group")
print("="*80)

key_features = ['accelerometer_Y_mean', 'accelerometer_Y_std', 'accelerometer_Y_max',
                'accelerometer_Z_mean', 'accelerometer_Z_std',
                'magnitude_mean', 'magnitude_std']

for feature in key_features:
    print(f"\n{feature}:")
    for group in ['misclassified', 'low_confidence', 'high_confidence']:
        group_data = all_features[all_features['group'] == group][feature]
        print(f"  {group:20s}: mean={group_data.mean():8.3f}, std={group_data.std():8.3f}")

# ============================================================================
# Visualization: Time series comparison
# ============================================================================
print("\n" + "="*80)
print("Creating Time Series Visualizations")
print("="*80)

# Plot misclassified samples
fig, axes = plt.subplots(4, 3, figsize=(15, 12))
fig.suptitle('Misclassified Stairs Samples (Predicted as Walking)', fontsize=16)

for idx, file_id in enumerate(misclassified_ids):
    sensor_df = train_combined[train_combined['id'] == file_id][
        ['time_step', 'accelerometer_X', 'accelerometer_Y', 'accelerometer_Z']
    ].sort_values('time_step')

    ax = axes[idx]
    ax[0].plot(sensor_df['time_step'], sensor_df['accelerometer_X'], 'r-', alpha=0.7, label='X')
    ax[0].plot(sensor_df['time_step'], sensor_df['accelerometer_Y'], 'g-', alpha=0.7, label='Y')
    ax[0].plot(sensor_df['time_step'], sensor_df['accelerometer_Z'], 'b-', alpha=0.7, label='Z')
    ax[0].set_title(f'{file_id}')
    ax[0].set_ylabel('Acceleration (m/s²)')
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    # Magnitude
    X = sensor_df['accelerometer_X'].values
    Y = sensor_df['accelerometer_Y'].values
    Z = sensor_df['accelerometer_Z'].values
    magnitude = np.sqrt(X**2 + Y**2 + Z**2)

    ax[1].plot(sensor_df['time_step'], magnitude, 'purple', alpha=0.7)
    ax[1].set_ylabel('Magnitude (m/s²)')
    ax[1].axhline(y=9.8, color='orange', linestyle='--', alpha=0.5, label='Gravity')
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    # Y-axis focus
    ax[2].plot(sensor_df['time_step'], sensor_df['accelerometer_Y'], 'g-', linewidth=2)
    ax[2].set_ylabel('Y Acceleration (m/s²)')
    ax[2].set_xlabel('Time Step')
    ax[2].grid(True, alpha=0.3)

plt.tight_layout()
plot_path = BASE_DIR / "stairs_misclassified_timeseries.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path}")

# Plot high confidence samples for comparison
sample_high_ids = np.random.choice(high_conf_stairs_ids, size=4, replace=False)
fig, axes = plt.subplots(4, 3, figsize=(15, 12))
fig.suptitle('High Confidence Stairs Samples (Correctly Classified)', fontsize=16)

for idx, file_id in enumerate(sample_high_ids):
    sensor_df = train_combined[train_combined['id'] == file_id][
        ['time_step', 'accelerometer_X', 'accelerometer_Y', 'accelerometer_Z']
    ].sort_values('time_step')

    ax = axes[idx]
    ax[0].plot(sensor_df['time_step'], sensor_df['accelerometer_X'], 'r-', alpha=0.7, label='X')
    ax[0].plot(sensor_df['time_step'], sensor_df['accelerometer_Y'], 'g-', alpha=0.7, label='Y')
    ax[0].plot(sensor_df['time_step'], sensor_df['accelerometer_Z'], 'b-', alpha=0.7, label='Z')
    ax[0].set_title(f'{file_id}')
    ax[0].set_ylabel('Acceleration (m/s²)')
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    # Magnitude
    X = sensor_df['accelerometer_X'].values
    Y = sensor_df['accelerometer_Y'].values
    Z = sensor_df['accelerometer_Z'].values
    magnitude = np.sqrt(X**2 + Y**2 + Z**2)

    ax[1].plot(sensor_df['time_step'], magnitude, 'purple', alpha=0.7)
    ax[1].set_ylabel('Magnitude (m/s²)')
    ax[1].axhline(y=9.8, color='orange', linestyle='--', alpha=0.5, label='Gravity')
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    # Y-axis focus
    ax[2].plot(sensor_df['time_step'], sensor_df['accelerometer_Y'], 'g-', linewidth=2)
    ax[2].set_ylabel('Y Acceleration (m/s²)')
    ax[2].set_xlabel('Time Step')
    ax[2].grid(True, alpha=0.3)

plt.tight_layout()
plot_path = BASE_DIR / "stairs_correct_timeseries.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path}")

# ============================================================================
# Feature distribution comparison
# ============================================================================
print("\n" + "="*80)
print("Creating Feature Distribution Plots")
print("="*80)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Feature Distribution: Misclassified vs High Confidence Stairs', fontsize=16)

features_to_plot = ['accelerometer_Y_mean', 'accelerometer_Y_std', 'accelerometer_Y_max',
                    'accelerometer_Z_std', 'magnitude_mean', 'magnitude_std']

for idx, feature in enumerate(features_to_plot):
    ax = axes[idx // 3, idx % 3]

    # Plot distributions
    for group, color in [('misclassified', 'red'), ('high_confidence', 'green')]:
        group_data = all_features[all_features['group'] == group][feature]
        ax.hist(group_data, alpha=0.5, label=group, color=color, bins=10)

    ax.set_xlabel(feature)
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = BASE_DIR / "stairs_feature_distributions.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path}")

# ============================================================================
# Export detailed analysis
# ============================================================================
print("\n" + "="*80)
print("Exporting Detailed Analysis")
print("="*80)

output_path = BASE_DIR / "stairs_feature_comparison.csv"
all_features.to_csv(output_path, index=False)
print(f"Saved feature comparison to: {output_path}")

# Summary report
print("\n" + "="*80)
print("Summary Report")
print("="*80)

print("\nMisclassified stairs samples show:")
print("  - 4 samples predicted as 'walking' instead of 'stairs'")
print("  - Stairs probability range: 18.3% - 43.3% (very low!)")
print("  - All confused with 'walking' class")

print("\nKey observations:")
misclass_y_std = all_features[all_features['group'] == 'misclassified']['accelerometer_Y_std'].mean()
highconf_y_std = all_features[all_features['group'] == 'high_confidence']['accelerometer_Y_std'].mean()
print(f"  - Y-axis std: misclassified={misclass_y_std:.3f}, high_conf={highconf_y_std:.3f}")

misclass_mag = all_features[all_features['group'] == 'misclassified']['magnitude_mean'].mean()
highconf_mag = all_features[all_features['group'] == 'high_confidence']['magnitude_mean'].mean()
print(f"  - Magnitude mean: misclassified={misclass_mag:.3f}, high_conf={highconf_mag:.3f}")

print("\nRecommendations:")
print("  1. Add stairs-specific features:")
print("     - Vertical (Z-axis) periodic patterns")
print("     - Y-Z axis correlation (different from walking)")
print("     - Magnitude variance over time")
print("  2. Class weighting or SMOTE for stairs class")
print("  3. Ensemble with models trained specifically on stairs vs non-stairs")

print("\n" + "="*80)
print("Analysis Complete!")
print("="*80)
