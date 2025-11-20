"""
LightGBM Baseline Model for Active Sensor Log Classification
Experiment: 20251120_05_baseline_lightgbm
"""

# ============================================================================
# Import
# ============================================================================
import os
import sys
import warnings
from pathlib import Path
import pickle
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    f1_score
)
import lightgbm as lgb

warnings.filterwarnings('ignore')

# ============================================================================
# Config
# ============================================================================
class Config:
    # Paths
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR.parent.parent / "data"
    PROCESSED_DIR = DATA_DIR / "processed"
    TRAIN_COMBINED = PROCESSED_DIR / "train_combined.csv"
    TEST_COMBINED = PROCESSED_DIR / "test_combined.csv"

    # Output paths
    PRED_DIR = BASE_DIR / "predictions"
    MODEL_DIR = BASE_DIR / "model"

    # CV settings
    N_SPLITS = 5
    RANDOM_STATE = 42

    # LightGBM parameters
    LGBM_PARAMS = {
        'objective': 'multiclass',
        'num_class': 4,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'min_child_samples': 20,
        'subsample': 0.8,
        'subsample_freq': 1,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': RANDOM_STATE,
        'verbose': -1,
        'n_jobs': -1
    }

    NUM_BOOST_ROUND = 1000
    EARLY_STOPPING_ROUNDS = 50
    VERBOSE_EVAL = 100

config = Config()

# ============================================================================
# Data Load
# ============================================================================
def load_all_data():
    """Load combined training and test data."""
    print("Loading combined training data...")
    train_combined = pd.read_csv(config.TRAIN_COMBINED)
    n_train_files = train_combined['id'].nunique()
    print(f"  Loaded {len(train_combined)} rows from {n_train_files} files")

    print("Loading combined test data...")
    test_combined = pd.read_csv(config.TEST_COMBINED)
    n_test_files = test_combined['id'].nunique()
    print(f"  Loaded {len(test_combined)} rows from {n_test_files} files")

    print(f"Data loading complete: {n_train_files} train files, {n_test_files} test files")
    return train_combined, test_combined

# ============================================================================
# Preprocess
# ============================================================================
def preprocess_sensor_data(df):
    """
    Preprocess sensor data.
    Currently no preprocessing needed (data is clean).
    """
    # Data is already validated in experiment 20251120_01
    # No missing values, correct shape, valid numeric values
    return df

# ============================================================================
# Feature Engineering
# ============================================================================
def extract_features(df, file_id):
    """Extract features from a single sensor data file."""
    features = {'id': file_id}

    # Axis names
    axes = ['accelerometer_X', 'accelerometer_Y', 'accelerometer_Z']

    # Statistical features for each axis
    for axis in axes:
        values = df[axis].values

        # Basic statistics
        features[f'{axis}_mean'] = np.mean(values)
        features[f'{axis}_std'] = np.std(values)
        features[f'{axis}_min'] = np.min(values)
        features[f'{axis}_max'] = np.max(values)
        features[f'{axis}_median'] = np.median(values)
        features[f'{axis}_q25'] = np.percentile(values, 25)
        features[f'{axis}_q75'] = np.percentile(values, 75)
        features[f'{axis}_iqr'] = features[f'{axis}_q75'] - features[f'{axis}_q25']

        # Range and variation
        features[f'{axis}_range'] = features[f'{axis}_max'] - features[f'{axis}_min']
        features[f'{axis}_cv'] = features[f'{axis}_std'] / (abs(features[f'{axis}_mean']) + 1e-8)

        # Higher order statistics
        features[f'{axis}_skew'] = pd.Series(values).skew()
        features[f'{axis}_kurtosis'] = pd.Series(values).kurtosis()

    # Magnitude of acceleration (coordinate-independent)
    X = df['accelerometer_X'].values
    Y = df['accelerometer_Y'].values
    Z = df['accelerometer_Z'].values
    magnitude = np.sqrt(X**2 + Y**2 + Z**2)

    features['magnitude_mean'] = np.mean(magnitude)
    features['magnitude_std'] = np.std(magnitude)
    features['magnitude_min'] = np.min(magnitude)
    features['magnitude_max'] = np.max(magnitude)
    features['magnitude_range'] = features['magnitude_max'] - features['magnitude_min']

    # Correlation between axes
    features['corr_XY'] = np.corrcoef(X, Y)[0, 1]
    features['corr_XZ'] = np.corrcoef(X, Z)[0, 1]
    features['corr_YZ'] = np.corrcoef(Y, Z)[0, 1]

    # Frequency domain features (FFT)
    for axis in axes:
        values = df[axis].values
        fft = np.fft.fft(values)
        fft_mag = np.abs(fft[:len(fft)//2])

        features[f'{axis}_fft_mean'] = np.mean(fft_mag)
        features[f'{axis}_fft_std'] = np.std(fft_mag)
        features[f'{axis}_fft_max'] = np.max(fft_mag)

        # Dominant frequency
        if len(fft_mag) > 0:
            features[f'{axis}_dominant_freq'] = np.argmax(fft_mag)
        else:
            features[f'{axis}_dominant_freq'] = 0

    # Dynamic acceleration statistics (magnitude - gravity)
    dynamic_accel = magnitude - 9.8
    features['dynamic_accel_mean'] = np.mean(dynamic_accel)
    features['dynamic_accel_std'] = np.std(dynamic_accel)
    features['dynamic_accel_max'] = np.max(dynamic_accel)

    return features

def create_feature_dataset(combined_df, is_train=True):
    """Create feature dataset from combined DataFrame."""
    n_files = combined_df['id'].nunique()
    print(f"Extracting features from {n_files} files...")

    features_list = []
    for idx, (file_id, group_df) in enumerate(combined_df.groupby('id')):
        # Extract sensor data columns only
        sensor_df = group_df[['accelerometer_X', 'accelerometer_Y', 'accelerometer_Z']].copy()

        # Preprocess
        sensor_df = preprocess_sensor_data(sensor_df)

        # Extract features
        features = extract_features(sensor_df, file_id)
        features_list.append(features)

        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{n_files} files")

    feature_df = pd.DataFrame(features_list)
    print(f"Feature extraction complete: shape {feature_df.shape}")

    return feature_df

# ============================================================================
# Data Split
# ============================================================================
def prepare_train_data(feature_df, train_combined):
    """Prepare training data with labels."""
    # Get unique id and class mapping from combined data
    train_labels = train_combined[['id', 'class']].drop_duplicates()

    # Merge with labels
    train_df = feature_df.merge(train_labels, on='id', how='left')

    # Encode labels
    label_mapping = {
        'running': 0,
        'walking': 1,
        'idle': 2,
        'stairs': 3
    }
    train_df['label'] = train_df['class'].map(label_mapping)

    # Separate features and labels
    X = train_df.drop(['id', 'class', 'label'], axis=1)
    y = train_df['label']
    ids = train_df['id']

    print(f"\nTraining data prepared:")
    print(f"  Features shape: {X.shape}")
    print(f"  Label distribution:\n{train_df['class'].value_counts()}")

    return X, y, ids, label_mapping

# ============================================================================
# Model
# ============================================================================
def create_lgb_datasets(X_train, y_train, X_val, y_val):
    """Create LightGBM datasets."""
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    return train_data, val_data

# ============================================================================
# Train
# ============================================================================
def train_with_cv(X, y, ids):
    """Train model with stratified K-fold cross validation."""
    print(f"\n{'='*80}")
    print(f"Starting {config.N_SPLITS}-Fold Cross Validation")
    print(f"{'='*80}")

    # Initialize arrays for OOF predictions
    oof_preds = np.zeros((len(X), 4))
    oof_labels = np.zeros(len(X))

    # Store models and fold indices
    models = []
    fold_indices = {'train': [], 'val': []}
    fold_scores = []

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'-'*80}")
        print(f"Fold {fold + 1}/{config.N_SPLITS}")
        print(f"{'-'*80}")

        # Split data
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        print(f"Train size: {len(X_train)}, Val size: {len(X_val)}")

        # Create LightGBM datasets
        train_data, val_data = create_lgb_datasets(X_train, y_train, X_val, y_val)

        # Train model
        model = lgb.train(
            config.LGBM_PARAMS,
            train_data,
            num_boost_round=config.NUM_BOOST_ROUND,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=config.EARLY_STOPPING_ROUNDS),
                lgb.log_evaluation(period=config.VERBOSE_EVAL)
            ]
        )

        # Predict on validation set
        val_preds = model.predict(X_val, num_iteration=model.best_iteration)
        oof_preds[val_idx] = val_preds
        oof_labels[val_idx] = y_val

        # Calculate fold score
        val_pred_labels = np.argmax(val_preds, axis=1)
        fold_logloss = log_loss(y_val, val_preds)
        fold_acc = accuracy_score(y_val, val_pred_labels)
        fold_f1 = f1_score(y_val, val_pred_labels, average='macro')
        fold_scores.append({
            'fold': fold + 1,
            'logloss': fold_logloss,
            'accuracy': fold_acc,
            'f1_macro': fold_f1
        })

        print(f"\nFold {fold + 1} Results:")
        print(f"  Log Loss: {fold_logloss:.6f}")
        print(f"  Accuracy: {fold_acc:.6f}")
        print(f"  F1 Macro: {fold_f1:.6f}")

        # Store model and indices
        models.append(model)
        fold_indices['train'].append(train_idx)
        fold_indices['val'].append(val_idx)

    # Calculate overall OOF score
    oof_pred_labels = np.argmax(oof_preds, axis=1)
    oof_logloss = log_loss(oof_labels, oof_preds)
    oof_acc = accuracy_score(oof_labels, oof_pred_labels)
    oof_f1 = f1_score(oof_labels, oof_pred_labels, average='macro')

    print(f"\n{'='*80}")
    print(f"Cross Validation Results")
    print(f"{'='*80}")
    for score in fold_scores:
        print(f"Fold {score['fold']}: F1 Macro = {score['f1_macro']:.6f}, "
              f"Accuracy = {score['accuracy']:.6f}, Log Loss = {score['logloss']:.6f}")

    print(f"\nOverall OOF Score:")
    print(f"  F1 Macro: {oof_f1:.6f}")
    print(f"  Accuracy: {oof_acc:.6f}")
    print(f"  Log Loss: {oof_logloss:.6f}")

    return models, oof_preds, oof_labels, fold_indices

def predict_test(models, X_test):
    """Predict on test data using ensemble of models."""
    print(f"\nPredicting on {len(X_test)} test samples...")

    test_preds = np.zeros((len(X_test), 4))

    for fold, model in enumerate(models):
        fold_preds = model.predict(X_test, num_iteration=model.best_iteration)
        test_preds += fold_preds
        print(f"  Fold {fold + 1} predictions done")

    # Average predictions across folds
    test_preds /= len(models)

    print("Test predictions complete")
    return test_preds

# ============================================================================
# Visualization and Analysis
# ============================================================================
def plot_confusion_matrix(y_true, y_pred, label_mapping, save_path):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    # Get class names in correct order
    inv_label_mapping = {v: k for k, v in label_mapping.items()}
    class_names = [inv_label_mapping[i] for i in range(len(inv_label_mapping))]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (OOF Predictions)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def plot_feature_importance(models, feature_names, save_path, top_n=30):
    """Plot average feature importance across folds."""
    # Calculate average importance
    importance_df = pd.DataFrame()
    for fold, model in enumerate(models):
        fold_importance = pd.DataFrame({
            'feature': feature_names,
            f'importance_fold_{fold}': model.feature_importance(importance_type='gain')
        })
        if importance_df.empty:
            importance_df = fold_importance
        else:
            importance_df = importance_df.merge(fold_importance, on='feature')

    # Calculate mean importance
    importance_cols = [col for col in importance_df.columns if col.startswith('importance_fold_')]
    importance_df['importance_mean'] = importance_df[importance_cols].mean(axis=1)
    importance_df['importance_std'] = importance_df[importance_cols].std(axis=1)

    # Sort and select top features
    importance_df = importance_df.sort_values('importance_mean', ascending=False).head(top_n)

    # Plot
    plt.figure(figsize=(10, 12))
    plt.barh(range(len(importance_df)), importance_df['importance_mean'],
             xerr=importance_df['importance_std'], align='center')
    plt.yticks(range(len(importance_df)), importance_df['feature'])
    plt.xlabel('Feature Importance (Gain)')
    plt.title(f'Top {top_n} Feature Importance (Average across folds)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Feature importance plot saved to {save_path}")

    return importance_df

def save_classification_report(y_true, y_pred, label_mapping, save_path):
    """Save classification report to file."""
    inv_label_mapping = {v: k for k, v in label_mapping.items()}
    target_names = [inv_label_mapping[i] for i in range(len(inv_label_mapping))]

    report = classification_report(y_true, y_pred, target_names=target_names)

    with open(save_path, 'w') as f:
        f.write("Classification Report (OOF Predictions)\n")
        f.write("="*80 + "\n\n")
        f.write(report)

    print(f"Classification report saved to {save_path}")

# ============================================================================
# Save Predictions
# ============================================================================
def save_predictions(oof_preds, oof_labels, test_preds, train_ids, test_ids,
                     label_mapping, fold_indices):
    """Save OOF and test predictions."""
    # Create output directory if not exists
    config.PRED_DIR.mkdir(parents=True, exist_ok=True)

    # Inverse label mapping
    inv_label_mapping = {v: k for k, v in label_mapping.items()}

    # OOF predictions
    oof_df = pd.DataFrame({
        'id': train_ids,
        'prob_running': oof_preds[:, 0],
        'prob_walking': oof_preds[:, 1],
        'prob_idle': oof_preds[:, 2],
        'prob_stairs': oof_preds[:, 3],
        'predicted_class': [inv_label_mapping[i] for i in np.argmax(oof_preds, axis=1)],
        'true_class': [inv_label_mapping[int(i)] for i in oof_labels]
    })
    oof_path = config.PRED_DIR / "oof.csv"
    oof_df.to_csv(oof_path, index=False)
    print(f"OOF predictions saved to {oof_path}")

    # Test predictions (probabilities)
    test_proba_df = pd.DataFrame({
        'id': test_ids,
        'prob_running': test_preds[:, 0],
        'prob_walking': test_preds[:, 1],
        'prob_idle': test_preds[:, 2],
        'prob_stairs': test_preds[:, 3]
    })
    test_proba_path = config.PRED_DIR / "test_proba.csv"
    test_proba_df.to_csv(test_proba_path, index=False)
    print(f"Test probabilities saved to {test_proba_path}")

    # Test predictions (submission format)
    test_submit_df = pd.DataFrame({
        'id': test_ids,
        'class': [inv_label_mapping[i] for i in np.argmax(test_preds, axis=1)]
    })
    test_submit_path = config.PRED_DIR / "test.csv"
    test_submit_df.to_csv(test_submit_path, index=False, header=False)
    print(f"Test submission file saved to {test_submit_path}")

    # Save fold indices
    fold_indices_path = config.PRED_DIR / "fold_indices.pkl"
    with open(fold_indices_path, 'wb') as f:
        pickle.dump(fold_indices, f)
    print(f"Fold indices saved to {fold_indices_path}")

def save_models(models):
    """Save trained models."""
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for fold, model in enumerate(models):
        model_path = config.MODEL_DIR / f"fold_{fold}.txt"
        model.save_model(str(model_path))
        print(f"Model fold {fold} saved to {model_path}")

# ============================================================================
# Main
# ============================================================================
def main(args):
    print("="*80)
    print("LightGBM Baseline Model Training")
    print("="*80)

    # Load data
    train_combined, test_combined = load_all_data()

    # Create features
    print("\n" + "="*80)
    print("Feature Engineering")
    print("="*80)
    train_features = create_feature_dataset(train_combined, is_train=True)
    test_features = create_feature_dataset(test_combined, is_train=False)

    # Prepare training data
    X, y, train_ids, label_mapping = prepare_train_data(train_features, train_combined)

    # Prepare test data
    test_ids = test_features['id'].values
    X_test = test_features.drop(['id'], axis=1)

    # Ensure test features match train features
    if not all(X.columns == X_test.columns):
        print("WARNING: Feature columns mismatch. Aligning test features...")
        X_test = X_test[X.columns]

    # Train with cross validation
    models, oof_preds, oof_labels, fold_indices = train_with_cv(X, y, train_ids)

    # Predict on test
    test_preds = predict_test(models, X_test)

    # Save predictions
    print("\n" + "="*80)
    print("Saving Predictions")
    print("="*80)
    save_predictions(oof_preds, oof_labels, test_preds, train_ids, test_ids,
                     label_mapping, fold_indices)

    # Save models
    if args.save_models:
        print("\n" + "="*80)
        print("Saving Models")
        print("="*80)
        save_models(models)

    # Visualization and analysis
    print("\n" + "="*80)
    print("Visualization and Analysis")
    print("="*80)

    # Confusion matrix
    oof_pred_labels = np.argmax(oof_preds, axis=1)
    cm_path = config.BASE_DIR / "confusion_matrix.png"
    plot_confusion_matrix(oof_labels, oof_pred_labels, label_mapping, cm_path)

    # Feature importance
    fi_path = config.BASE_DIR / "feature_importance.png"
    importance_df = plot_feature_importance(models, X.columns.tolist(), fi_path, top_n=30)

    # Save feature importance to CSV
    fi_csv_path = config.BASE_DIR / "feature_importance.csv"
    importance_df.to_csv(fi_csv_path, index=False)
    print(f"Feature importance saved to {fi_csv_path}")

    # Classification report
    report_path = config.BASE_DIR / "classification_report.txt"
    save_classification_report(oof_labels, oof_pred_labels, label_mapping, report_path)

    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LightGBM baseline model")
    parser.add_argument('--save-models', action='store_true',
                        help='Save trained models (default: False)')
    args = parser.parse_args()

    main(args)
