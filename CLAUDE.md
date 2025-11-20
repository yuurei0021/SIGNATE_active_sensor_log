# CLAUDE.md

## プロジェクト概要
このリポジトリはMLコンペ「活動センサーログによる動作分類」のデータ分析およびモデル作成を行う

### 目的
与えられたTrainデータを用いてモデルを作成し、Testデータに対して最も精度の良いモデルを作成すること

### 評価指標
- **F1 Macro（マクロ平均F1スコア）**: コンペティションの公式評価指標
- 各クラス（running, walking, idle, stairs）のF1スコアを算出し、その平均を取る
- クラス不均衡に対して公平な評価が可能
- モデル開発・評価時は必ずF1 Macroを主要指標として使用すること

## 環境

- **Python**: 3.11以上
- **パッケージマネージャ**: uv
- **依存関係**: `pyproject.toml` に定義（numpy 1.x系、pandas、scikit-learn、xgboost、lightgbm、matplotlib、seaborn）
- **実行**: `uv run python experiments/YYYYMMDD_NN_実験名/main.py`

## ファイル構成

```
SIGNATE_active_sensor_log/
├── data/
│   ├── raw/
│   │   ├── test/           # テストデータ (test_00000.csv ~ test_xxxxx.csv)
│   │   ├── train/          # 訓練データ (train_00000.csv ~ train_xxxxx.csv)
│   │   ├── train_master.csv # 訓練データのラベル情報
│   │   └── sample_submit.csv # 提出ファイルサンプル
│   └── processed/          # 前処理済みデータ
│       ├── prepare_data.py      # データ結合スクリプト
│       ├── train_combined.csv   # 結合済み訓練データ
│       └── test_combined.csv    # 結合済みテストデータ
├── experiments/            # 実験用フォルダ
│   └── YYYYMMDD_NN_実験名/ # 実験ごとにフォルダを作成（NNは連番01, 02...）
│       ├── main.py         # 実験用スクリプト（AI Agent対応）
│       ├── predictions/    # 予測結果
│       │   ├── oof.csv            # OOF予測（ID + 確率 + 予測クラス）
│       │   ├── test_proba.csv     # テスト予測確率
│       │   ├── test.csv           # テスト予測クラス（提出用）
│       │   └── fold_indices.pkl   # CV fold情報（オプション）
│       ├── model/          # 学習済みモデル
│       └── README.md       # 実験の詳細記録
├── .venv/                  # 仮想環境（gitignore対象）
├── pyproject.toml          # プロジェクト設定・依存関係
├── .python-version         # Python バージョン指定
├── .gitignore              # Git除外設定
└── CLAUDE.md              # このファイル
```

## データ形式

### train_master.csv
```csv
id,class
train_00000,running
train_00001,running
train_00003,walking
```
- 4,523件、クラス: `running`, `walking`, `idle`, `stairs`

### train/*.csv
```csv
accelerometer_X,accelerometer_Y,accelerometer_Z
8.695741,-1.149217,-6.325481
-2.485181,6.770803,-1.656788
...
```
- 3軸加速度センサー、30行/ファイル（0.5秒間隔、15秒間）
- **座標系**（実験20251120_04で特定）: X, Y, Z軸は**スマートフォンに固定された座標系**。動作によってスマートフォンの向きが異なる（idle: Z軸が上向き、walking/stairs: ポケット内で安定、running: 向きが不安定）。重力加速度は常に存在するが、動作中は動的加速度により隠される

### test/*.csv
- 訓練データと同じ形式の3軸加速度センサーデータ

### sample_submit.csv
```csv
test_00000,stairs
test_00001,stairs
test_00002,walking
```
- 提出ファイルの形式: `test_id,predicted_class`

### 結合済みデータ（推奨）

**重要**: 毎回数千のファイルを読み込むのは非効率なため、事前に結合データを使用することを推奨

#### train_combined.csv
- 全訓練ファイルを結合したデータ（135,690行）
- カラム: `accelerometer_X`, `accelerometer_Y`, `accelerometer_Z`, `id`, `time_step`, `class`
- `id`: ファイルID（train_00000など）
- `time_step`: ファイル内の行番号（0-29）
- 作成方法: `uv run python data/processed/prepare_data.py`

#### test_combined.csv
- 全テストファイルを結合したデータ（58,170行）
- カラム: `accelerometer_X`, `accelerometer_Y`, `accelerometer_Z`, `id`, `time_step`
- 作成方法: 同上

## 実験管理ルール

### 基本原則
- 各実験は独立したフォルダで管理し、再現性と追跡可能性を確保する
- **CLAUDE.md更新ルール**: リポジトリに変更を加えた際、その情報をAI Agentが今後の作業で考慮する必要がある場合は、本ファイル（CLAUDE.md）を更新すること

### フォルダ命名規則
- **形式**: `YYYYMMDD_NN_実験名`（NNは連番: 01, 02, 03...）
- **例**: `20251120_01_data_validation`, `20251120_02_baseline_rf`, `20251121_01_feature_engineering`
- **場所**: `experiments/`ディレクトリ配下
- **同日複数実験**: 連番により実行順序を明確化

### 実験フォルダ構成
各実験フォルダには以下を含める：

1. **main.py** (必須)
   - データ読み込みから予測まで一連の処理を実装
   - AI Agentによる管理・実行を想定したPythonスクリプト
   - コマンドライン引数でパラメータ調整可能にする

2. **predictions/** (必須)
   - `oof.csv`: OOF予測結果（id, 各クラス確率, 予測クラス）
   - `test_proba.csv`: テスト予測確率（id, 各クラス確率）
   - `test.csv`: テスト予測クラス（提出用、sample_submit.csvと同じフォーマット）
   - `fold_indices.pkl`: CV fold情報（オプション、再現性確保のため）

3. **model/** (必要に応じて)
   - 学習済みモデルファイル (.pkl, .h5, .pth等)
   - CV使用時は各foldのモデルを保存（例: `fold_0.pkl`, `fold_1.pkl`, ...）

4. **README.md** (推奨)
   - 実験の詳細な記録
   - ハイパーパラメータ
   - 精度指標（OOF score, CV scoreなど）
   - 気づきや改善点

### 実験履歴の記録
- 各実験完了後、本ファイル（CLAUDE.md）の「実験履歴」セクションに要約を追加
- 見出しは実験フォルダ名と一致させる（例: `### 20251120_01_data_validation`）
- 目的、アプローチ、結果、次のステップを記録

### コミットルール
- 実験フォルダ作成時にコミット
- 実験完了・中断時にコミット
- コミットメッセージ例: `Add experiment: 20251120_01_data_validation`
- pushはユーザーから明示的に指示されたときのみ実行すること

## 実験履歴

### 20251120_01_data_validation
**目的**: 全trainとtestデータがモデル入力可能な状態かを検証し、前処理の必要性を判断
**アプローチ**: 全4,523 trainファイルと1,939 testファイルに対して、欠損値・データ形状・数値型・無限大値を検証
**結果**: 全データが有効（前処理不要）。クラス分布: running(52.2%), walking(29.0%), idle(16.4%), stairs(2.4%)
**次のステップ**: ベースラインモデル構築（クラス不均衡対策として層化サンプリングやクラス重み付けを検討）

---

### 20251120_02_visualization
**目的**: センサーデータの特性を理解し、各動作クラス間のデータ分布の違いを可視化
**アプローチ**: 基本統計量、時系列パターン、クラス別の平均値・標準偏差を可視化（4種類のグラフを生成）
**結果**: running/walkingは周期的変動、idleは変動小、stairsはY軸に特徴的パターン。標準偏差が重要な特徴量と判明
**次のステップ**: 周波数領域特徴量（FFT）の抽出、時系列統計量（歪度・尖度）の計算

---

### 20251120_03_axis_analysis
**目的**: X, Y, Z軸がそれぞれ上下・左右・前後のどの方向に相当するかをデータから特定
**アプローチ**: Idle状態での重力加速度検出、各動作での軸別平均値・標準偏差分析、時系列パターン比較
**結果**: Z軸=上下（idle時Z≈9.0）、Y軸=前後（walking/runningで最大変動）、X軸=左右。センサー向き: 前方向が負、上方向が正
**次のステップ**: 重力補正（Z-9.8）、Y軸の周波数解析、軸間相関を特徴量として活用
**注**: この結論は20251120_04で修正された（固定座標系ではなく、スマートフォン固定座標系）

---

### 20251120_04_gravity_investigation
**目的**: idle時は重力加速度が明確だが他の動作時に見えない理由を解明。X, Y, Z軸が動作ごとに方向が異なるかを検証
**アプローチ**: 合成加速度の大きさ分析、サンプルごとの平均加速度ベクトル分析、動的加速度の分離、時系列での加速度変化
**結果**: 座標系はスマートフォンに固定されており、動作によって端末の向きが異なる。重力は常に存在するが、running時の動的加速度（≈15.5 m/s²）が重力（9.8 m/s²）を上回り、重力成分が「見えなく」なる。合成加速度の大きさ（idle≈9.8, walking≈12.9, running≈17.0）は動作強度を表す有効な特徴量
**次のステップ**: 座標系に依存しない特徴量設計（加速度の大きさ、軸間相関、周波数成分、動的加速度の統計量）

---

### 20251120_05_baseline_lightgbm
**目的**: LightGBMによるベースラインモデルの構築。時間領域・周波数領域の特徴量を用いた高精度モデルの実現
**アプローチ**:
- 効率化のため、事前に全データを結合（train_combined.csv, test_combined.csv）
- 59次元の特徴量（各軸の統計量、FFT、合成加速度、軸間相関、動的加速度）
- 5-Fold Stratified CV、Early Stopping使用
**結果**: **OOF F1 Macro = 0.9950**（99.50%）、**Public LB F1 Macro = 1.0000**（100.00%）。Y軸（前後方向）の特徴量が最重要。テストデータに対して完璧な予測を達成
**次のステップ**: Private LBに備えてstairsクラスの改善が必要（OOFで4サンプル誤分類）

---

### 20251120_06_stairs_analysis
**目的**: stairsクラスの誤分類原因を特定し、Private LBでの完璧な精度達成に向けた改善策を提案
**アプローチ**:
- OOF予測結果の詳細分析（4件の誤分類、14件の低信頼度予測を特定）
- センサーデータの特徴比較（誤分類 vs 正解サンプル）
- 時系列パターンの可視化と統計分析
**結果**:
- 誤分類の原因特定：stairsとwalkingの類似性、Z軸の周期性が捉えきれていない、サンプル数不足（110件 vs 1,311件）
- 改善策を3段階で提案：Phase 1（stairs特有特徴量+クラス重み）、Phase 2（SMOTE+閾値調整）、Phase 3（アンサンブル+NN）
**次のステップ**: Phase 1の実装

---

### 20251120_07_stairs_improvement
**目的**: Phase 1改善策の実装によりstairsクラスの予測精度を向上
**アプローチ**:
- stairs特有の特徴量追加（+13次元）：Z軸の周期性（ピーク・谷・サイクル）、Y-Z軸の相関パターン（移動窓）、変動比率（Z/Y、X/Y、Z/Mag）
- クラス重み調整：stairsに5倍の重み付与
- ベースラインと同じ5-Fold Stratified CV
**結果**: **OOF F1 Macro = 0.9988**（99.88%）、**Private LB F1 Macro = 1.0000**（100.00%）。ベースラインから+0.0038向上。誤分類4件→1件（75%削減）。Public/Private両方で完璧なスコアを達成
**次のステップ**: コンペティション完了。完璧な予測を達成したため、これ以上の改善は不要

---
