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

def analyze_axis_statistics(master_df, n_samples=100):
    print("Analyzing axis statistics...")

    classes = master_df['class'].unique()
    stats = {cls: {'mean_x': [], 'mean_y': [], 'mean_z': [],
                   'std_x': [], 'std_y': [], 'std_z': [],
                   'abs_mean_x': [], 'abs_mean_y': [], 'abs_mean_z': []}
             for cls in classes}

    for cls in classes:
        cls_data = master_df[master_df['class'] == cls].iloc[:n_samples]

        for _, row in cls_data.iterrows():
            data = load_sensor_data(row['id'])

            stats[cls]['mean_x'].append(data['accelerometer_X'].mean())
            stats[cls]['mean_y'].append(data['accelerometer_Y'].mean())
            stats[cls]['mean_z'].append(data['accelerometer_Z'].mean())

            stats[cls]['abs_mean_x'].append(data['accelerometer_X'].abs().mean())
            stats[cls]['abs_mean_y'].append(data['accelerometer_Y'].abs().mean())
            stats[cls]['abs_mean_z'].append(data['accelerometer_Z'].abs().mean())

            stats[cls]['std_x'].append(data['accelerometer_X'].std())
            stats[cls]['std_y'].append(data['accelerometer_Y'].std())
            stats[cls]['std_z'].append(data['accelerometer_Z'].std())

    # Summary table
    print("\n" + "="*80)
    print("AXIS STATISTICS SUMMARY (mean ± std)")
    print("="*80)
    print(f"{'Class':<15} {'X axis':<25} {'Y axis':<25} {'Z axis':<25}")
    print("-"*80)

    for cls in classes:
        mean_x = np.mean(stats[cls]['mean_x'])
        std_mean_x = np.std(stats[cls]['mean_x'])

        mean_y = np.mean(stats[cls]['mean_y'])
        std_mean_y = np.std(stats[cls]['mean_y'])

        mean_z = np.mean(stats[cls]['mean_z'])
        std_mean_z = np.std(stats[cls]['mean_z'])

        print(f"{cls:<15} {mean_x:>6.2f} ± {std_mean_x:<5.2f}       "
              f"{mean_y:>6.2f} ± {std_mean_y:<5.2f}       "
              f"{mean_z:>6.2f} ± {std_mean_z:<5.2f}")

    print("\n" + "="*80)
    print("AXIS VARIATION (std within sample, averaged across samples)")
    print("="*80)
    print(f"{'Class':<15} {'X std':<10} {'Y std':<10} {'Z std':<10}")
    print("-"*80)

    for cls in classes:
        avg_std_x = np.mean(stats[cls]['std_x'])
        avg_std_y = np.mean(stats[cls]['std_y'])
        avg_std_z = np.mean(stats[cls]['std_z'])

        print(f"{cls:<15} {avg_std_x:<10.2f} {avg_std_y:<10.2f} {avg_std_z:<10.2f}")

    return stats

def plot_axis_comparison(stats):
    print("\nGenerating axis comparison plots...")

    classes = list(stats.keys())

    # Plot 1: Mean values by class and axis
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Mean values comparison
    for i, axis in enumerate(['x', 'y', 'z']):
        ax = axes[i // 2, i % 2]
        data_to_plot = [stats[cls][f'mean_{axis}'] for cls in classes]

        positions = np.arange(len(classes))
        bp = ax.boxplot(data_to_plot, positions=positions, patch_artist=True)

        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')

        ax.set_xticks(positions)
        ax.set_xticklabels(classes, rotation=45)
        ax.set_title(f'Mean Acceleration {axis.upper()} by Class')
        ax.set_ylabel(f'Mean Acceleration {axis.upper()}')
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.axhline(y=9.8, color='g', linestyle='--', alpha=0.5, label='g=9.8')
        ax.axhline(y=-9.8, color='g', linestyle='--', alpha=0.5)
        ax.legend()

    # Std comparison
    ax = axes[1, 1]
    x_pos = np.arange(len(classes))
    width = 0.25

    avg_std = {axis: [np.mean(stats[cls][f'std_{axis}']) for cls in classes]
               for axis in ['x', 'y', 'z']}

    ax.bar(x_pos - width, avg_std['x'], width, label='X', alpha=0.8)
    ax.bar(x_pos, avg_std['y'], width, label='Y', alpha=0.8)
    ax.bar(x_pos + width, avg_std['z'], width, label='Z', alpha=0.8)

    ax.set_xlabel('Class')
    ax.set_ylabel('Average Std Deviation')
    ax.set_title('Standard Deviation by Class and Axis')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(classes, rotation=45)
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_axis_statistics.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR / '01_axis_statistics.png'}")

def analyze_idle_gravity(master_df, n_samples=50):
    print("\nAnalyzing idle state for gravity detection...")

    idle_data = master_df[master_df['class'] == 'idle'].iloc[:n_samples]

    all_x, all_y, all_z = [], [], []

    for _, row in idle_data.iterrows():
        data = load_sensor_data(row['id'])
        all_x.extend(data['accelerometer_X'].values)
        all_y.extend(data['accelerometer_Y'].values)
        all_z.extend(data['accelerometer_Z'].values)

    print(f"\nIdle state statistics ({n_samples} samples, {len(all_x)} total points):")
    print(f"X axis: mean={np.mean(all_x):.3f}, std={np.std(all_x):.3f}")
    print(f"Y axis: mean={np.mean(all_y):.3f}, std={np.std(all_y):.3f}")
    print(f"Z axis: mean={np.mean(all_z):.3f}, std={np.std(all_z):.3f}")

    # Histogram of idle values
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(all_x, bins=50, alpha=0.7)
    axes[0].set_title('Idle - X axis distribution')
    axes[0].set_xlabel('Acceleration X')
    axes[0].axvline(x=9.8, color='r', linestyle='--', label='g=9.8')
    axes[0].axvline(x=-9.8, color='r', linestyle='--')
    axes[0].legend()

    axes[1].hist(all_y, bins=50, alpha=0.7)
    axes[1].set_title('Idle - Y axis distribution')
    axes[1].set_xlabel('Acceleration Y')
    axes[1].axvline(x=9.8, color='r', linestyle='--', label='g=9.8')
    axes[1].axvline(x=-9.8, color='r', linestyle='--')
    axes[1].legend()

    axes[2].hist(all_z, bins=50, alpha=0.7)
    axes[2].set_title('Idle - Z axis distribution')
    axes[2].set_xlabel('Acceleration Z')
    axes[2].axvline(x=9.8, color='r', linestyle='--', label='g=9.8')
    axes[2].axvline(x=-9.8, color='r', linestyle='--')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_idle_gravity_analysis.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR / '02_idle_gravity_analysis.png'}")

def analyze_motion_patterns(master_df):
    print("\nAnalyzing motion patterns...")

    motion_classes = ['running', 'walking', 'stairs']

    fig, axes = plt.subplots(len(motion_classes), 3, figsize=(18, len(motion_classes) * 4))

    for i, cls in enumerate(motion_classes):
        sample_ids = master_df[master_df['class'] == cls].iloc[:3]['id'].values

        for j, (axis_name, color) in enumerate([('accelerometer_X', 'blue'),
                                                  ('accelerometer_Y', 'green'),
                                                  ('accelerometer_Z', 'red')]):
            ax = axes[i, j]

            for sample_id in sample_ids:
                data = load_sensor_data(sample_id)
                ax.plot(data[axis_name], alpha=0.6, color=color)

            ax.set_title(f'{cls.capitalize()} - {axis_name.split("_")[1]} axis')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Acceleration')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_motion_patterns.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR / '03_motion_patterns.png'}")

def main():
    print("Axis Direction Analysis")
    print("=" * 60)

    OUTPUT_DIR.mkdir(exist_ok=True)

    master_df = load_train_master()

    stats = analyze_axis_statistics(master_df, n_samples=100)
    plot_axis_comparison(stats)
    analyze_idle_gravity(master_df, n_samples=50)
    analyze_motion_patterns(master_df)

    print("\n" + "=" * 60)
    print("Analysis complete. Check the output/ directory.")

if __name__ == "__main__":
    main()
