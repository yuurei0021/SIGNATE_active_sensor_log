# Stairs Class Analysis Report

## 実行日
2025-11-20

## 概要
LightGBMベースラインモデルのOOF予測において、stairsクラスの予測精度を詳細に分析。Privateデータでの性能向上を目指し、誤分類サンプルと低信頼度予測を特定・分析。

## 分析結果

### 1. 全体の精度

| Class   | Accuracy | Correct | Total |
|---------|----------|---------|-------|
| running | 100.00%  | 2361    | 2361  |
| walking | 100.00%  | 1311    | 1311  |
| idle    | 100.00%  | 741     | 741   |
| **stairs** | **96.36%** | **106** | **110** |

- **4サンプルが誤分類**（すべてwalkingと予測）
- **14サンプルが予測確率95%未満**

### 2. 誤分類されたサンプル

| ID | True | Predicted | P(running) | P(walking) | P(idle) | P(stairs) |
|----|------|-----------|------------|------------|---------|-----------|
| train_02397 | stairs | walking | 0.162% | **81.4%** | 0.138% | 18.3% |
| train_02999 | stairs | walking | 0.404% | **72.6%** | 0.335% | 27.0% |
| train_00857 | stairs | walking | 2.138% | **58.4%** | 1.342% | 38.1% |
| train_00970 | stairs | walking | 0.187% | **56.3%** | 0.155% | 43.3% |

**共通点**: すべてwalkingとの境界が曖昧

### 3. 低信頼度サンプル（予測確率 < 95%）

10サンプルが正しくstairsと分類されたが、確率が95%未満：

| ID | P(stairs) | Status |
|----|-----------|--------|
| train_02840 | 63.5% | 要注意 |
| train_04236 | 81.9% | 要注意 |
| train_02285 | 82.9% | 要注意 |
| train_03260 | 83.0% | 要注意 |
| train_00691 | 87.4% | やや低 |
| train_02021 | 88.5% | やや低 |
| train_01451 | 91.8% | やや低 |
| train_04134 | 91.9% | やや低 |
| train_02036 | 94.2% | やや低 |
| train_03970 | 94.7% | やや低 |

### 4. センサーデータの特徴比較

#### Y軸（前後方向）の統計量

| Group | Mean | Std | Max |
|-------|------|-----|-----|
| 誤分類 | -9.031 | **4.847** | -0.061 |
| 低信頼度 | -10.253 | **5.695** | 1.047 |
| 高信頼度 | -9.460 | **4.508** | -0.510 |

**観察**:
- 誤分類・低信頼度サンプルはY軸の標準偏差が高い
- これはwalkingの特徴に近い（周期的な前後の動き）

#### Z軸（上下方向）の統計量

| Group | Mean | Std |
|-------|------|-----|
| 誤分類 | **-0.901** | 5.442 |
| 低信頼度 | -1.709 | 5.681 |
| 高信頼度 | **-2.027** | 4.640 |

**観察**:
- 誤分類サンプルはZ軸の平均値が高い（0に近い）
- 高信頼度サンプルは明確な負の値を示す

#### 合成加速度

| Group | Mean | Std |
|-------|------|-----|
| 誤分類 | 11.405 | 5.007 |
| 低信頼度 | **12.624** | **6.064** |
| 高信頼度 | 11.231 | 5.103 |

**観察**:
- 低信頼度サンプルは動的加速度が大きい
- walkingとの境界が曖昧になる要因

## 問題の根本原因

### 1. stairsとwalkingの類似性
- 両方とも周期的な動き
- Y軸（前後）の変動パターンが類似
- 階段の登降でも前後の動きが含まれる

### 2. サンプル数の不足
- stairs: 110サンプル（全体の2.4%）
- walking: 1311サンプル（全体の29.0%）
- 約12倍の差 → モデルがwalkingに偏る

### 3. 特徴量の不足
- 現在の特徴量はwalkingとの区別が不十分
- Z軸（上下）の周期的パターンを捉えきれていない
- stairsに特化した特徴量がない

## 改善策

### 優先度 高: 即座に実装可能

#### 1. stairs特有の特徴量を追加

**Z軸の周期性検出**:
```python
# Z軸のピーク検出（上下の繰り返し）
from scipy.signal import find_peaks
peaks, _ = find_peaks(Z_values)
n_peaks = len(peaks)
peak_interval = np.mean(np.diff(peaks)) if len(peaks) > 1 else 0
```

**Y-Z軸の相関パターン**:
```python
# walkingとstairsで異なる相関パターン
corr_YZ = np.corrcoef(Y_values, Z_values)[0, 1]

# 移動窓での相関変化
window_size = 10
rolling_corr = [np.corrcoef(Y[i:i+window_size], Z[i:i+window_size])[0,1]
                for i in range(len(Y)-window_size)]
corr_variance = np.var(rolling_corr)
```

**Z軸の変動比率**:
```python
# stairsではZ軸の変動がY軸に対して相対的に大きい
z_to_y_ratio = np.std(Z_values) / (np.std(Y_values) + 1e-8)
```

#### 2. クラス重み調整

```python
# LightGBMのクラス重みを調整
class_weights = {
    0: 1.0,    # running
    1: 1.0,    # walking
    2: 1.0,    # idle
    3: 5.0,    # stairs (重みを5倍に)
}

# または自動計算
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=[0,1,2,3], y=y_train)
```

#### 3. 予測確率の閾値調整

```python
# stairsの判定閾値を下げる（現在は単純なargmax）
# 例: stairs確率が40%以上かつ他のクラスより高ければstairsと判定
def adjusted_prediction(probs):
    if probs[3] > 0.40 and probs[3] == max(probs):
        return 3  # stairs
    return np.argmax(probs)
```

### 優先度 中: やや時間がかかる

#### 4. データ拡張（SMOTE）

```python
from imblearn.over_sampling import SMOTE

# stairsサンプルを合成生成
smote = SMOTE(sampling_strategy={3: 300}, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

#### 5. 2段階モデル

```python
# Stage 1: stairs vs non-stairs (binary classification)
# Stage 2: non-stairsを running/walking/idle に分類

# Stage 1で高精度にstairsを特定
binary_model = LGBMClassifier(...)
binary_model.fit(X_train, y_binary)  # 1=stairs, 0=other

# Stage 2で残りを分類
```

### 優先度 低: 時間がかかる

#### 6. アンサンブル

- XGBoost、CatBoost など他のモデルと組み合わせ
- stairsに特化したモデルを別途作成してアンサンブル

#### 7. ニューラルネットワーク

- 1D-CNN: 時系列パターンを直接学習
- LSTM: 長期依存関係を捉える

## 推奨実装順序

### Phase 1: 即座に実装（1-2時間）
1. stairs特有の特徴量追加（Z軸周期性、Y-Z相関、Z/Y比率）
2. クラス重み調整（重み5-10倍）

### Phase 2: 中期実装（3-5時間）
3. 予測確率の閾値調整
4. SMOTE によるデータ拡張
5. 2段階モデルの検討

### Phase 3: 長期実装（必要に応じて）
6. アンサンブル手法
7. ニューラルネットワーク

## 期待される効果

### Phase 1実装後の予測
- 誤分類4サンプル → 1-2サンプルに削減
- 低信頼度14サンプル → 5-8サンプルに削減
- **OOF F1 Macro: 0.9950 → 0.9980以上**
- **Private LB: さらなる向上が期待**

### Phase 2実装後の予測
- 誤分類 → ゼロを目指す
- すべてのstairsサンプルで95%以上の確率
- **OOF F1 Macro: 0.9990以上**

## 次のアクション

1. **新しい実験フォルダ作成**: `20251120_06_stairs_improvement`
2. **Phase 1の特徴量を実装**
3. **モデル再訓練とOOF評価**
4. **効果を確認後、Phase 2へ進む**

## 分析ファイル

- `stairs_analysis_prob_distribution.png`: クラス別の予測確率分布
- `stairs_analysis_stairs_prob_hist.png`: stairs確率のヒストグラム
- `stairs_misclassified_timeseries.png`: 誤分類サンプルの時系列データ
- `stairs_correct_timeseries.png`: 正しく分類されたサンプルの時系列データ
- `stairs_feature_distributions.png`: 特徴量分布の比較
- `stairs_problematic_samples.csv`: 問題のあるサンプルのリスト
- `stairs_feature_comparison.csv`: 特徴量の詳細比較

---

**結論**: stairsクラスの改善は十分に実現可能。特にPhase 1の特徴量追加とクラス重み調整により、大きな効果が期待できる。
