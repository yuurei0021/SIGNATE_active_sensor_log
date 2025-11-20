# Experiment: 20251120_06_stairs_analysis

## 概要
LightGBMベースラインモデル（実験20251120_05）のOOF予測結果を詳細に分析し、stairsクラスの誤分類原因を特定。Private LBでの性能向上に向けた改善策を提案。

## 目的
- stairsクラスの誤分類サンプルを特定
- 低信頼度予測の原因を分析
- センサーデータの特徴を比較
- 具体的な改善策を提案

## アプローチ

### 1. 基本分析（analysis_stairs.py）
- OOF予測結果の読み込みと集計
- クラス別の精度確認
- 誤分類サンプルの特定
- 予測確率の分布分析
- 可視化（確率分布、ヒストグラム）

### 2. 詳細分析（analysis_stairs_detailed.py）
- 問題サンプルのセンサーデータ読み込み
- 誤分類・低信頼度・高信頼度サンプルの特徴量比較
- 時系列データの可視化
- 特徴量分布の比較
- 統計的分析

## 結果

### 問題の特定

#### 誤分類サンプル: 4件（精度96.36%）

| ID | 真のクラス | 予測 | Stairs確率 | Walking確率 |
|----|-----------|------|------------|-------------|
| train_02397 | stairs | walking | 18.3% | 81.4% |
| train_02999 | stairs | walking | 27.0% | 72.6% |
| train_00857 | stairs | walking | 38.1% | 58.4% |
| train_00970 | stairs | walking | 43.3% | 56.3% |

**共通点**: すべてwalkingと混同

#### 低信頼度サンプル: 14件（予測確率 < 95%）

最も低い確率:
- train_02840: 63.5%
- train_04236: 81.9%
- train_02285: 82.9%

### センサーデータの特徴比較

#### Y軸（前後方向）の統計量

| グループ | 平均 | 標準偏差 | 最大値 |
|----------|------|----------|--------|
| 誤分類 | -9.031 | **4.847** | -0.061 |
| 低信頼度 | -10.253 | **5.695** | 1.047 |
| 高信頼度 | -9.460 | **4.508** | -0.510 |

**観察**:
- 誤分類・低信頼度サンプルはY軸の標準偏差が高い
- walkingの特徴（周期的な前後の動き）に類似

#### Z軸（上下方向）の統計量

| グループ | 平均 | 標準偏差 |
|----------|------|----------|
| 誤分類 | **-0.901** | 5.442 |
| 低信頼度 | -1.709 | 5.681 |
| 高信頼度 | **-2.027** | 4.640 |

**観察**:
- 誤分類サンプルはZ軸の平均値が高い（0に近い）
- 高信頼度サンプルは明確な負の値

#### 合成加速度

| グループ | 平均 | 標準偏差 |
|----------|------|----------|
| 誤分類 | 11.405 | 5.007 |
| 低信頼度 | **12.624** | **6.064** |
| 高信頼度 | 11.231 | 5.103 |

### 根本原因

1. **stairsとwalkingの類似性**
   - 両方とも周期的な動き
   - Y軸の変動パターンが類似
   - 階段の登降でも前後の動きが含まれる

2. **サンプル数の不足**
   - stairs: 110サンプル（2.4%）
   - walking: 1,311サンプル（29.0%）
   - 約12倍の差 → モデルがwalkingに偏る

3. **特徴量の不足**
   - Z軸（上下）の周期的パターンを捉えきれていない
   - stairsに特化した特徴量がない

## 改善策

### 優先度 高: 即座に実装可能

#### 1. stairs特有の特徴量を追加

**Z軸の周期性検出**:
- ピーク数、ピーク間隔
- Z軸のFFTでの卓越周波数

**Y-Z軸の相関パターン**:
- 移動窓での相関変化
- walkingとは異なる相関パターン

**Z/Y変動比率**:
- stairsではZ軸の変動が相対的に大きい
- `z_to_y_ratio = std(Z) / std(Y)`

#### 2. クラス重み調整

```python
class_weights = {
    0: 1.0,    # running
    1: 1.0,    # walking
    2: 1.0,    # idle
    3: 5.0,    # stairs (重みを5倍に)
}
```

#### 3. 予測確率の閾値調整

- stairsの判定閾値を下げる
- 例: stairs確率が40%以上かつ最大値ならstairsと判定

### 優先度 中

4. データ拡張（SMOTE）
5. 2段階モデル（stairs vs 非stairs → 3クラス分類）

### 優先度 低

6. アンサンブル（XGBoost、CatBoost）
7. ニューラルネットワーク（1D-CNN、LSTM）

## 期待される効果

### Phase 1実装後
- 誤分類: 4 → 1-2サンプル
- 低信頼度: 14 → 5-8サンプル
- **OOF F1 Macro: 0.9950 → 0.9980+**

### Phase 2実装後
- 誤分類: → ゼロ
- すべてのstairsサンプルで95%以上の確率
- **OOF F1 Macro: 0.9990+**

## 出力ファイル

### 分析スクリプト
- `analysis_stairs.py` - 基本分析
- `analysis_stairs_detailed.py` - 詳細分析

### レポート・データ
- `STAIRS_ANALYSIS_REPORT.md` - 詳細レポート
- `stairs_problematic_samples.csv` - 問題サンプルリスト（14件）
- `stairs_feature_comparison.csv` - 特徴量比較データ

### 可視化
- `stairs_analysis_prob_distribution.png` - クラス別予測確率分布
- `stairs_analysis_stairs_prob_hist.png` - stairs確率ヒストグラム
- `stairs_misclassified_timeseries.png` - 誤分類4サンプルの時系列データ
- `stairs_correct_timeseries.png` - 正解サンプルの時系列データ
- `stairs_feature_distributions.png` - 特徴量分布比較

## 次のステップ

1. **実験20251120_07_stairs_improvement**を作成
2. Phase 1の改善策を実装
   - stairs特有の特徴量追加
   - クラス重み調整
3. モデル再訓練とOOF評価
4. 効果を確認後、Phase 2へ進む

## 実行方法

```bash
# 基本分析
uv run python experiments/20251120_06_stairs_analysis/analysis_stairs.py

# 詳細分析
uv run python experiments/20251120_06_stairs_analysis/analysis_stairs_detailed.py
```

## 関連実験

- **20251120_05_baseline_lightgbm**: ベースラインモデル（OOF F1 Macro: 0.9950）
- **20251120_07_stairs_improvement**: 改善策の実装（予定）
