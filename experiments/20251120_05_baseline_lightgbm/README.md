# Experiment: 20251120_05_baseline_lightgbm

## 概要
LightGBMを使用したベースラインモデルの構築。

## 目的
- センサーデータから抽出した特徴量を用いてLightGBMモデルを学習
- 5-Fold Stratified Cross Validationによる性能評価
- テストデータに対する予測を生成

## 評価指標
- **F1 Macro**: メインの評価指標（コンペティションの評価基準）
- Accuracy: 参考指標
- Log Loss: 学習時の最適化指標

## アプローチ

### コード構成
1. **Import**: 必要なライブラリのインポート
2. **Config**: ハイパーパラメータとパスの設定
3. **Data Load**: 訓練・テストデータの読み込み
4. **Preprocess**: データの前処理（現在は不要）
5. **Feature Engineering**: 特徴量抽出
6. **Data Split**: 層化K分割交差検証
7. **Model**: LightGBMモデルの定義
8. **Train**: モデルの学習
9. **Visualization**: 混同行列、特徴量重要度、分類レポート

### 特徴量
**時間領域の統計量（各軸X, Y, Z）**:
- 平均、標準偏差、最小値、最大値、中央値
- 25/75パーセンタイル、四分位範囲（IQR）
- レンジ、変動係数
- 歪度、尖度

**合成加速度の大きさ（座標系非依存）**:
- magnitude = sqrt(X^2 + Y^2 + Z^2)
- 平均、標準偏差、最小値、最大値、レンジ

**軸間相関**:
- X-Y, X-Z, Y-Z相関係数

**周波数領域の特徴量（FFT）**:
- FFTの平均、標準偏差、最大値
- 支配的な周波数成分

**動的加速度の統計量**:
- 重力補正後の加速度（magnitude - 9.8）の平均、標準偏差、最大値

### モデルパラメータ
```python
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
    'random_state': 42,
    'n_jobs': -1
}
```

- **num_boost_round**: 1000
- **early_stopping_rounds**: 50
- **N_SPLITS**: 5（層化K分割交差検証）

## 実行方法
```bash
# 基本実行
uv run python experiments/20251120_05_baseline_lightgbm/main.py

# モデルを保存する場合
uv run python experiments/20251120_05_baseline_lightgbm/main.py --save-models
```

## 結果

### CV Score
- **OOF F1 Macro**: **0.994990** (99.50%)
- OOF Accuracy: 0.999116 (99.91%)
- OOF Log Loss: 0.001860

### Fold別スコア
| Fold | F1 Macro | Accuracy | Log Loss |
|------|----------|----------|----------|
| 1    | 1.000000 | 1.000000 | 0.000418 |
| 2    | 1.000000 | 1.000000 | 0.000859 |
| 3    | 0.993710 | 0.998895 | 0.001786 |
| 4    | 0.987145 | 0.997788 | 0.003621 |
| 5    | 0.993710 | 0.998894 | 0.002617 |

### クラス別性能（OOF）
| Class   | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| running | 1.00      | 1.00   | 1.00     | 2361    |
| walking | 1.00      | 1.00   | 1.00     | 1311    |
| idle    | 1.00      | 1.00   | 1.00     | 741     |
| stairs  | 1.00      | 0.96   | 0.98     | 110     |

**注**: stairsクラスのrecallが0.96とやや低いが、サンプル数が少ない（110件）ことが影響していると考えられる。

### 重要な特徴量Top 10
1. **accelerometer_Y_max** (15158.5) - Y軸の最大値
2. **accelerometer_Z_min** (11160.4) - Z軸の最小値
3. **accelerometer_Y_median** (8981.9) - Y軸の中央値
4. **accelerometer_Y_q75** (7465.2) - Y軸の75パーセンタイル
5. **accelerometer_Y_q25** (6086.2) - Y軸の25パーセンタイル
6. **accelerometer_Y_mean** (2764.3) - Y軸の平均値
7. **accelerometer_Y_min** (1784.2) - Y軸の最小値
8. **accelerometer_Y_std** (1742.5) - Y軸の標準偏差
9. **accelerometer_Z_q25** (1580.1) - Z軸の25パーセンタイル
10. **accelerometer_Z_q75** (1391.6) - Z軸の75パーセンタイル

**知見**: Y軸（前後方向）の特徴量が圧倒的に重要。これは実験20251120_02の可視化結果と一致。

## 出力ファイル
- `predictions/oof.csv`: OOF予測（ID + 各クラス確率 + 予測クラス + 真のクラス）
- `predictions/test_proba.csv`: テスト予測確率
- `predictions/test.csv`: テスト予測クラス（提出用）
- `predictions/fold_indices.pkl`: CV fold情報
- `model/fold_0.txt ~ fold_4.txt`: 学習済みモデル（--save-modelsオプション使用時）
- `confusion_matrix.png`: 混同行列
- `feature_importance.png`: 特徴量重要度
- `feature_importance.csv`: 特徴量重要度データ
- `classification_report.txt`: 分類レポート

## 分析と考察

### 成功要因
1. **適切な特徴量設計**: 時間領域・周波数領域・座標系非依存の特徴量を組み合わせた
2. **Y軸の重要性**: 前後方向の動きが動作分類に最も重要（歩行・走行の周期的な動き）
3. **層化サンプリング**: クラス不均衡に対応した適切なCV戦略

### 課題
1. **stairsクラスの分類**: サンプル数が少ない（110件）ため、recallが0.96とやや低い
2. **過学習の可能性**: 訓練データでの性能が非常に高いため、テストデータでの汎化性能を確認する必要がある

## 次のステップ
1. **stairsクラスの改善**:
   - データ拡張（SMOTE等）
   - クラス重み調整
   - stairs特有の特徴量（Z軸の周期的変動など）の追加
2. **モデルの多様化**:
   - XGBoost、CatBoostなど他のGBDTモデルとのアンサンブル
   - ニューラルネットワーク（1D-CNN、LSTM等）の検討
3. **特徴量の追加**:
   - ウェーブレット変換
   - 時系列の勾配・加速度の変化率
   - 周波数領域のより詳細な分析（ピーク検出等）
