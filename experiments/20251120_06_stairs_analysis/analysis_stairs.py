"""
Analysis of stairs class predictions
Focus on misclassified samples and low-confidence predictions
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
BASE_DIR = Path(__file__).parent
PRED_DIR = BASE_DIR / "predictions"
DATA_DIR = BASE_DIR.parent.parent / "data" / "processed"

# Load OOF predictions
print("="*80)
print("Stairs Class Prediction Analysis")
print("="*80)

print("\nLoading OOF predictions...")
oof_df = pd.read_csv(PRED_DIR / "oof.csv")
print(f"OOF predictions loaded: {len(oof_df)} samples")

print("\nLabel distribution:")
print(oof_df['true_class'].value_counts())

# ============================================================================
# 1. Overall accuracy by class
# ============================================================================
print("\n" + "="*80)
print("1. Overall Accuracy by Class")
print("="*80)

for class_name in ['running', 'walking', 'idle', 'stairs']:
    class_df = oof_df[oof_df['true_class'] == class_name]
    correct = (class_df['predicted_class'] == class_df['true_class']).sum()
    total = len(class_df)
    accuracy = correct / total if total > 0 else 0
    print(f"{class_name:>10s}: {correct:4d}/{total:4d} correct ({accuracy:.2%})")

# ============================================================================
# 2. Misclassified samples
# ============================================================================
print("\n" + "="*80)
print("2. Misclassified Samples")
print("="*80)

misclassified = oof_df[oof_df['predicted_class'] != oof_df['true_class']]
print(f"\nTotal misclassified: {len(misclassified)}")

if len(misclassified) > 0:
    print("\nMisclassified samples by true class:")
    print(misclassified.groupby('true_class')['predicted_class'].value_counts())

    print("\nMisclassified sample details:")
    print(misclassified[['id', 'true_class', 'predicted_class',
                         'prob_running', 'prob_walking', 'prob_idle', 'prob_stairs']])
else:
    print("No misclassified samples!")

# ============================================================================
# 3. Stairs class detailed analysis
# ============================================================================
print("\n" + "="*80)
print("3. Stairs Class Detailed Analysis")
print("="*80)

stairs_true = oof_df[oof_df['true_class'] == 'stairs'].copy()
print(f"\nTotal stairs samples: {len(stairs_true)}")

# Prediction distribution for true stairs
print("\nPrediction distribution for true stairs:")
print(stairs_true['predicted_class'].value_counts())

# Stairs probability statistics
print(f"\nStairs probability statistics (for true stairs):")
print(f"  Mean:   {stairs_true['prob_stairs'].mean():.6f}")
print(f"  Median: {stairs_true['prob_stairs'].median():.6f}")
print(f"  Min:    {stairs_true['prob_stairs'].min():.6f}")
print(f"  Max:    {stairs_true['prob_stairs'].max():.6f}")
print(f"  Std:    {stairs_true['prob_stairs'].std():.6f}")

# Low confidence stairs predictions
threshold = 0.95
low_conf_stairs = stairs_true[stairs_true['prob_stairs'] < threshold]
print(f"\nLow confidence stairs predictions (prob < {threshold}):")
print(f"  Count: {len(low_conf_stairs)}")

if len(low_conf_stairs) > 0:
    print("\nLow confidence stairs samples:")
    print(low_conf_stairs[['id', 'predicted_class', 'prob_running', 'prob_walking',
                            'prob_idle', 'prob_stairs']].sort_values('prob_stairs'))

# Misclassified stairs
stairs_misclassified = stairs_true[stairs_true['predicted_class'] != 'stairs']
print(f"\nMisclassified stairs samples: {len(stairs_misclassified)}")

if len(stairs_misclassified) > 0:
    print("\nMisclassified stairs details:")
    for idx, row in stairs_misclassified.iterrows():
        print(f"\nID: {row['id']}")
        print(f"  Predicted as: {row['predicted_class']}")
        print(f"  Probabilities:")
        print(f"    running: {row['prob_running']:.6f}")
        print(f"    walking: {row['prob_walking']:.6f}")
        print(f"    idle:    {row['prob_idle']:.6f}")
        print(f"    stairs:  {row['prob_stairs']:.6f}")

# ============================================================================
# 4. Predicted as stairs (but might be wrong)
# ============================================================================
print("\n" + "="*80)
print("4. Predicted as Stairs Analysis")
print("="*80)

predicted_stairs = oof_df[oof_df['predicted_class'] == 'stairs'].copy()
print(f"\nTotal predicted as stairs: {len(predicted_stairs)}")

# Correct vs incorrect
correct_stairs = predicted_stairs[predicted_stairs['true_class'] == 'stairs']
incorrect_stairs = predicted_stairs[predicted_stairs['true_class'] != 'stairs']

print(f"  Correct: {len(correct_stairs)}")
print(f"  Incorrect: {len(incorrect_stairs)}")

if len(incorrect_stairs) > 0:
    print("\nIncorrectly predicted as stairs:")
    print(incorrect_stairs[['id', 'true_class', 'prob_running', 'prob_walking',
                             'prob_idle', 'prob_stairs']])

# Low confidence stairs predictions (might be misclassified on test set)
threshold_pred = 0.90
low_conf_pred_stairs = predicted_stairs[predicted_stairs['prob_stairs'] < threshold_pred]
print(f"\nLow confidence 'predicted as stairs' (prob < {threshold_pred}):")
print(f"  Count: {len(low_conf_pred_stairs)}")

if len(low_conf_pred_stairs) > 0:
    print(low_conf_pred_stairs[['id', 'true_class', 'prob_running', 'prob_walking',
                                 'prob_idle', 'prob_stairs']].sort_values('prob_stairs'))

# ============================================================================
# 5. Visualization
# ============================================================================
print("\n" + "="*80)
print("5. Creating Visualizations")
print("="*80)

# Plot 1: Probability distribution for each true class
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Prediction Probability Distributions by True Class', fontsize=16)

for idx, class_name in enumerate(['running', 'walking', 'idle', 'stairs']):
    ax = axes[idx // 2, idx % 2]
    class_df = oof_df[oof_df['true_class'] == class_name]

    # Plot probability distribution for all classes
    probs = class_df[['prob_running', 'prob_walking', 'prob_idle', 'prob_stairs']]
    probs.columns = ['Running', 'Walking', 'Idle', 'Stairs']

    probs.boxplot(ax=ax)
    ax.set_title(f'True Class: {class_name.capitalize()} (n={len(class_df)})')
    ax.set_ylabel('Prediction Probability')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = BASE_DIR / "stairs_analysis_prob_distribution.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path}")

# Plot 2: Stairs probability distribution for true stairs
fig, ax = plt.subplots(figsize=(10, 6))
stairs_true['prob_stairs'].hist(bins=50, ax=ax, edgecolor='black')
ax.axvline(x=0.95, color='red', linestyle='--', label='95% threshold')
ax.set_xlabel('Stairs Probability')
ax.set_ylabel('Frequency')
ax.set_title(f'Stairs Probability Distribution (True Stairs Samples, n={len(stairs_true)})')
ax.legend()
ax.grid(True, alpha=0.3)

plot_path = BASE_DIR / "stairs_analysis_stairs_prob_hist.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {plot_path}")

# ============================================================================
# 6. Export problematic samples
# ============================================================================
print("\n" + "="*80)
print("6. Exporting Problematic Samples")
print("="*80)

# Combine misclassified and low confidence samples
problematic_ids = set()

if len(stairs_misclassified) > 0:
    problematic_ids.update(stairs_misclassified['id'].tolist())

if len(low_conf_stairs) > 0:
    problematic_ids.update(low_conf_stairs['id'].tolist())

problematic_df = oof_df[oof_df['id'].isin(problematic_ids)].copy()
problematic_df = problematic_df.sort_values('prob_stairs')

if len(problematic_df) > 0:
    output_path = BASE_DIR / "stairs_problematic_samples.csv"
    problematic_df.to_csv(output_path, index=False)
    print(f"Exported {len(problematic_df)} problematic samples to: {output_path}")

    print("\nProblematic samples summary:")
    print(problematic_df[['id', 'true_class', 'predicted_class',
                          'prob_running', 'prob_walking', 'prob_idle', 'prob_stairs']])
else:
    print("No problematic samples found!")

print("\n" + "="*80)
print("Analysis Complete!")
print("="*80)
