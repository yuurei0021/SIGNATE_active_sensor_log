# CLAUDE.md

## プロジェクト概要
このリポジトリはMLコンペ「活動センサーログによる動作分類」のデータ分析およびモデル作成を行う

### 目的
与えられたTrainデータを用いてモデルを作成し、Testデータに対して最も精度の良いモデルを作成すること

## 環境

- **Python**: 3.11以上
- **パッケージマネージャ**: uv
- **依存関係**: `pyproject.toml` に定義（numpy 1.x系、pandas、scikit-learn、xgboost、lightgbm、matplotlib、seaborn）
- **実行**: `uv run python experiments/YYYYMMDD_NN_実験名/main.py`

## ファイル構成

```
SIGNATE_active_sensor_log/
├── data/
│   └── raw/
│       ├── test/           # テストデータ (test_00000.csv ~ test_xxxxx.csv)
│       ├── train/          # 訓練データ (train_00000.csv ~ train_xxxxx.csv)
│       ├── train_master.csv # 訓練データのラベル情報
│       └── sample_submit.csv # 提出ファイルサンプル
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

### test/*.csv
- 訓練データと同じ形式の3軸加速度センサーデータ

### sample_submit.csv
```csv
test_00000,stairs
test_00001,stairs
test_00002,walking
```
- 提出ファイルの形式: `test_id,predicted_class`

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
