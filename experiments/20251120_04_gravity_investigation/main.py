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

def load_train_master():
    return pd.read_csv(DATA_DIR / "train_master.csv")

def load_sensor_data(file_id):
    file_path = TRAIN_DIR / f"{file_id}.csv"
    return pd.read_csv(file_path)

def analyze_acceleration_magnitude(master_df, n_samples=100):
    print("Analyzing acceleration magnitude...")

    classes = master_df['class'].unique()
    magnitude_stats = {cls: {'magnitudes': [], 'mean_mag': [], 'std_mag': []}
                       for cls in classes}

    for cls in classes:
        cls_data = master_df[master_df['class'] == cls].iloc[:n_samples]

        for _, row in cls_data.iterrows():
            data = load_sensor_data(row['id'])

            magnitude = np.sqrt(data['accelerometer_X']**2 +
                              data['accelerometer_Y']**2 +
                              data['accelerometer_Z']**2)

            magnitude_stats[cls]['magnitudes'].extend(magnitude.values)
            magnitude_stats[cls]['mean_mag'].append(magnitude.mean())
            magnitude_stats[cls]['std_mag'].append(magnitude.std())

    print("\n" + "="*60)
    print("ACCELERATION MAGNITUDE STATISTICS")
    print("="*60)
    print(f"{'Class':<15} {'Mean Magnitude':<20} {'Std of Magnitude':<20}")
    print("-"*60)

    for cls in classes:
        mean_of_means = np.mean(magnitude_stats[cls]['mean_mag'])
        mean_of_stds = np.mean(magnitude_stats[cls]['std_mag'])
        print(f"{cls:<15} {mean_of_means:<20.3f} {mean_of_stds:<20.3f}")

    # Plot magnitude distributions
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    for i, cls in enumerate(classes):
        ax = axes[i // 2, i % 2]
        ax.hist(magnitude_stats[cls]['magnitudes'], bins=50, alpha=0.7, edgecolor='black')
        ax.set_title(f'{cls.capitalize()} - Acceleration Magnitude Distribution')
        ax.set_xlabel('Magnitude (m/s²)')
        ax.set_ylabel('Frequency')
        ax.axvline(x=9.8, color='r', linestyle='--', label='g=9.8', linewidth=2)
        ax.legend()

        mean_val = np.mean(magnitude_stats[cls]['magnitudes'])
        ax.text(0.7, 0.9, f'Mean: {mean_val:.2f}',
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_magnitude_distribution.png", dpi=150)
    plt.close()
    print(f"\nSaved: {OUTPUT_DIR / '01_magnitude_distribution.png'}")

    return magnitude_stats

def analyze_gravity_direction_per_sample(master_df, n_samples=20):
    print("\nAnalyzing gravity direction for each sample...")

    classes = master_df['class'].unique()

    fig, axes = plt.subplots(len(classes), 1, figsize=(15, len(classes) * 4))
    if len(classes) == 1:
        axes = [axes]

    for cls_idx, cls in enumerate(classes):
        cls_data = master_df[master_df['class'] == cls].iloc[:n_samples]

        mean_vectors = []
        for _, row in cls_data.iterrows():
            data = load_sensor_data(row['id'])
            mean_x = data['accelerometer_X'].mean()
            mean_y = data['accelerometer_Y'].mean()
            mean_z = data['accelerometer_Z'].mean()
            mean_vectors.append([mean_x, mean_y, mean_z])

        mean_vectors = np.array(mean_vectors)

        ax = axes[cls_idx]

        # 3D scatter of mean acceleration vectors
        x_means = mean_vectors[:, 0]
        y_means = mean_vectors[:, 1]
        z_means = mean_vectors[:, 2]

        scatter = ax.scatter(x_means, y_means, c=z_means, cmap='viridis', s=100, alpha=0.6)
        ax.set_xlabel('Mean X acceleration')
        ax.set_ylabel('Mean Y acceleration')
        ax.set_title(f'{cls.capitalize()} - Mean Acceleration Vectors (Color=Z value)')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)

        plt.colorbar(scatter, ax=ax, label='Mean Z')

        print(f"\n{cls.capitalize()}:")
        print(f"  X range: [{x_means.min():.2f}, {x_means.max():.2f}]")
        print(f"  Y range: [{y_means.min():.2f}, {y_means.max():.2f}]")
        print(f"  Z range: [{z_means.min():.2f}, {z_means.max():.2f}]")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_gravity_direction_scatter.png", dpi=150)
    plt.close()
    print(f"\nSaved: {OUTPUT_DIR / '02_gravity_direction_scatter.png'}")

def analyze_magnitude_time_series(master_df):
    print("\nAnalyzing magnitude time series...")

    classes = master_df['class'].unique()

    fig, axes = plt.subplots(len(classes), 1, figsize=(15, len(classes) * 3))
    if len(classes) == 1:
        axes = [axes]

    for cls_idx, cls in enumerate(classes):
        sample_ids = master_df[master_df['class'] == cls].iloc[:5]['id'].values

        ax = axes[cls_idx]

        for sample_id in sample_ids:
            data = load_sensor_data(sample_id)
            magnitude = np.sqrt(data['accelerometer_X']**2 +
                              data['accelerometer_Y']**2 +
                              data['accelerometer_Z']**2)
            ax.plot(magnitude, alpha=0.6)

        ax.axhline(y=9.8, color='r', linestyle='--', label='g=9.8', linewidth=2)
        ax.set_title(f'{cls.capitalize()} - Magnitude Time Series')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Acceleration Magnitude (m/s²)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_magnitude_time_series.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR / '03_magnitude_time_series.png'}")

def analyze_dynamic_vs_gravity(master_df, n_samples=50):
    print("\nAnalyzing dynamic acceleration vs gravity...")

    classes = master_df['class'].unique()

    # For each class, calculate dynamic acceleration (deviation from mean)
    results = {}

    for cls in classes:
        cls_data = master_df[master_df['class'] == cls].iloc[:n_samples]

        total_magnitudes = []
        dynamic_magnitudes = []

        for _, row in cls_data.iterrows():
            data = load_sensor_data(row['id'])

            # Total acceleration magnitude
            total_mag = np.sqrt(data['accelerometer_X']**2 +
                              data['accelerometer_Y']**2 +
                              data['accelerometer_Z']**2)

            # Mean acceleration (approximates gravity component)
            mean_x = data['accelerometer_X'].mean()
            mean_y = data['accelerometer_Y'].mean()
            mean_z = data['accelerometer_Z'].mean()

            # Dynamic acceleration (deviation from mean)
            dynamic_x = data['accelerometer_X'] - mean_x
            dynamic_y = data['accelerometer_Y'] - mean_y
            dynamic_z = data['accelerometer_Z'] - mean_z

            dynamic_mag = np.sqrt(dynamic_x**2 + dynamic_y**2 + dynamic_z**2)

            total_magnitudes.extend(total_mag.values)
            dynamic_magnitudes.extend(dynamic_mag.values)

        results[cls] = {
            'total_mag': total_magnitudes,
            'dynamic_mag': dynamic_magnitudes
        }

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    for i, cls in enumerate(classes):
        ax = axes[i // 2, i % 2]

        ax.hist(results[cls]['total_mag'], bins=50, alpha=0.5, label='Total', color='blue')
        ax.hist(results[cls]['dynamic_mag'], bins=50, alpha=0.5, label='Dynamic', color='orange')

        ax.set_title(f'{cls.capitalize()} - Total vs Dynamic Acceleration')
        ax.set_xlabel('Magnitude (m/s²)')
        ax.set_ylabel('Frequency')
        ax.axvline(x=9.8, color='r', linestyle='--', linewidth=2, label='g=9.8')
        ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_dynamic_vs_total.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR / '04_dynamic_vs_total.png'}")

    print("\n" + "="*60)
    print("DYNAMIC ACCELERATION ANALYSIS")
    print("="*60)
    print(f"{'Class':<15} {'Mean Total':<15} {'Mean Dynamic':<15}")
    print("-"*60)

    for cls in classes:
        mean_total = np.mean(results[cls]['total_mag'])
        mean_dynamic = np.mean(results[cls]['dynamic_mag'])
        print(f"{cls:<15} {mean_total:<15.3f} {mean_dynamic:<15.3f}")

def main():
    print("Gravity Investigation - Detailed Analysis")
    print("=" * 60)

    OUTPUT_DIR.mkdir(exist_ok=True)

    master_df = load_train_master()

    magnitude_stats = analyze_acceleration_magnitude(master_df, n_samples=100)
    analyze_gravity_direction_per_sample(master_df, n_samples=30)
    analyze_magnitude_time_series(master_df)
    analyze_dynamic_vs_gravity(master_df, n_samples=50)

    print("\n" + "=" * 60)
    print("Analysis complete. Check the output/ directory.")

if __name__ == "__main__":
    main()
