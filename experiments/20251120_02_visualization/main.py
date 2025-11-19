import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = REPO_ROOT / "data" / "raw"
TRAIN_DIR = DATA_DIR / "train"
OUTPUT_DIR = Path(__file__).parent / "output"

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def load_train_master():
    return pd.read_csv(DATA_DIR / "train_master.csv")

def load_sensor_data(file_id):
    file_path = TRAIN_DIR / f"{file_id}.csv"
    return pd.read_csv(file_path)

def plot_basic_visualization(master_df):
    print("Generating basic visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Class distribution
    class_counts = master_df['class'].value_counts()
    axes[0, 0].bar(class_counts.index, class_counts.values)
    axes[0, 0].set_title('Class Distribution')
    axes[0, 0].set_xlabel('Class')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Sample time series from each class
    sample_data = {}
    for cls in master_df['class'].unique():
        sample_id = master_df[master_df['class'] == cls].iloc[0]['id']
        sample_data[cls] = load_sensor_data(sample_id)

    # Plot sample time series for X axis
    axes[0, 1].set_title('Sample Time Series (X axis)')
    for cls, data in sample_data.items():
        axes[0, 1].plot(data['accelerometer_X'], label=cls, alpha=0.7)
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Acceleration X')
    axes[0, 1].legend()

    # Distribution of all axes (combined sample)
    all_x = []
    all_y = []
    all_z = []

    for i in range(min(100, len(master_df))):
        data = load_sensor_data(master_df.iloc[i]['id'])
        all_x.extend(data['accelerometer_X'].values)
        all_y.extend(data['accelerometer_Y'].values)
        all_z.extend(data['accelerometer_Z'].values)

    axes[1, 0].hist(all_x, bins=50, alpha=0.5, label='X')
    axes[1, 0].hist(all_y, bins=50, alpha=0.5, label='Y')
    axes[1, 0].hist(all_z, bins=50, alpha=0.5, label='Z')
    axes[1, 0].set_title('Acceleration Distribution (100 samples)')
    axes[1, 0].set_xlabel('Acceleration')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()

    # Acceleration magnitude distribution
    magnitudes = [np.sqrt(x**2 + y**2 + z**2) for x, y, z in zip(all_x, all_y, all_z)]
    axes[1, 1].hist(magnitudes, bins=50, alpha=0.7)
    axes[1, 1].set_title('Acceleration Magnitude Distribution')
    axes[1, 1].set_xlabel('Magnitude')
    axes[1, 1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_basic_visualization.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR / '01_basic_visualization.png'}")

def plot_class_comparison(master_df):
    print("Generating class comparison visualizations...")

    classes = master_df['class'].unique()
    n_samples_per_class = 50

    class_stats = {cls: {'mean_x': [], 'mean_y': [], 'mean_z': [],
                          'std_x': [], 'std_y': [], 'std_z': [],
                          'magnitude_mean': [], 'magnitude_std': []}
                   for cls in classes}

    for cls in classes:
        cls_data = master_df[master_df['class'] == cls].iloc[:n_samples_per_class]

        for _, row in cls_data.iterrows():
            data = load_sensor_data(row['id'])

            class_stats[cls]['mean_x'].append(data['accelerometer_X'].mean())
            class_stats[cls]['mean_y'].append(data['accelerometer_Y'].mean())
            class_stats[cls]['mean_z'].append(data['accelerometer_Z'].mean())
            class_stats[cls]['std_x'].append(data['accelerometer_X'].std())
            class_stats[cls]['std_y'].append(data['accelerometer_Y'].std())
            class_stats[cls]['std_z'].append(data['accelerometer_Z'].std())

            magnitude = np.sqrt(data['accelerometer_X']**2 +
                              data['accelerometer_Y']**2 +
                              data['accelerometer_Z']**2)
            class_stats[cls]['magnitude_mean'].append(magnitude.mean())
            class_stats[cls]['magnitude_std'].append(magnitude.std())

    # Plot 1: Mean acceleration by class
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    for i, axis in enumerate(['x', 'y', 'z']):
        ax = axes[i // 2, i % 2]
        data_to_plot = [class_stats[cls][f'mean_{axis}'] for cls in classes]
        ax.boxplot(data_to_plot, labels=classes)
        ax.set_title(f'Mean Acceleration {axis.upper()} by Class')
        ax.set_ylabel(f'Mean Acceleration {axis.upper()}')
        ax.tick_params(axis='x', rotation=45)

    # Magnitude mean
    data_to_plot = [class_stats[cls]['magnitude_mean'] for cls in classes]
    axes[1, 1].boxplot(data_to_plot, labels=classes)
    axes[1, 1].set_title('Mean Acceleration Magnitude by Class')
    axes[1, 1].set_ylabel('Mean Magnitude')
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_class_mean_comparison.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR / '02_class_mean_comparison.png'}")

    # Plot 2: Std deviation by class
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    for i, axis in enumerate(['x', 'y', 'z']):
        ax = axes[i // 2, i % 2]
        data_to_plot = [class_stats[cls][f'std_{axis}'] for cls in classes]
        ax.boxplot(data_to_plot, labels=classes)
        ax.set_title(f'Std Deviation {axis.upper()} by Class')
        ax.set_ylabel(f'Std Deviation {axis.upper()}')
        ax.tick_params(axis='x', rotation=45)

    # Magnitude std
    data_to_plot = [class_stats[cls]['magnitude_std'] for cls in classes]
    axes[1, 1].boxplot(data_to_plot, labels=classes)
    axes[1, 1].set_title('Std Deviation of Magnitude by Class')
    axes[1, 1].set_ylabel('Std Deviation')
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_class_std_comparison.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR / '03_class_std_comparison.png'}")

def plot_class_time_series(master_df):
    print("Generating class-specific time series...")

    classes = master_df['class'].unique()
    fig, axes = plt.subplots(len(classes), 3, figsize=(18, len(classes) * 3))

    for i, cls in enumerate(classes):
        sample_ids = master_df[master_df['class'] == cls].iloc[:5]['id'].values

        for j, axis_name in enumerate(['accelerometer_X', 'accelerometer_Y', 'accelerometer_Z']):
            ax = axes[i, j] if len(classes) > 1 else axes[j]

            for sample_id in sample_ids:
                data = load_sensor_data(sample_id)
                ax.plot(data[axis_name], alpha=0.5)

            ax.set_title(f'{cls.capitalize()} - {axis_name.split("_")[1]} axis')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Acceleration')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_class_time_series.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR / '04_class_time_series.png'}")

def main():
    print("Data Visualization Experiment")
    print("=" * 60)

    OUTPUT_DIR.mkdir(exist_ok=True)

    master_df = load_train_master()
    print(f"Loaded {len(master_df)} training samples")
    print(f"Classes: {master_df['class'].unique()}")

    plot_basic_visualization(master_df)
    plot_class_comparison(master_df)
    plot_class_time_series(master_df)

    print("\n" + "=" * 60)
    print("Visualization complete. Check the output/ directory.")

if __name__ == "__main__":
    main()
