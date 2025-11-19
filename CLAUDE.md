# CLAUDE.md

## プロジェクト概要
このリポジトリはMLコンペ「活動センサーログによる動作分類」のデータ分析およびモデル作成を行う

### 目的
与えられたTrainデータを用いてモデルを作成し、Testデータに対して最も精度の良いモデルを作成すること

## 環境

- **Python**: 3.11以上
- **パッケージマネージャ**: uv
- **依存関係**: `pyproject.toml` に定義（numpy 1.x系、pandas、scikit-learn、xgboost、lightgbm、matplotlib、seaborn）
- **実行**: `uv run python experiments/YYYYMMDD_実験名/main.py`

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
│   └── YYYYMMDD_実験名/   # 実験ごとにフォルダを作成
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

### 訓練データ
- **train_master.csv**: `id,class` - 4,523件、クラス: `running`, `walking`, `idle`, `stairs`
- **train/*.csv**: `accelerometer_X,accelerometer_Y,accelerometer_Z` - 3軸加速度センサー、30行/ファイル（0.5秒間隔、15秒間）

### テストデータ
- **test/*.csv**: 訓練データと同じ形式の3軸加速度センサーデータ

### 提出形式
- **sample_submit.csv**: `test_id,predicted_class` - 提出ファイルの形式

## 実験管理ルール

### 基本原則
各実験は独立したフォルダで管理し、再現性と追跡可能性を確保する。

### フォルダ命名規則
- **形式**: `YYYYMMDD_実験名`
- **例**: `20250119_baseline_rf`, `20250120_deep_lstm`
- **場所**: `experiments/`ディレクトリ配下

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
- 日付、目的、主な内容、精度、次のステップを記録

### コミットルール
- 実験フォルダ作成時にコミット
- 実験完了・中断時にコミット
- コミットメッセージ例: `Add experiment: 20250119_baseline_rf`

## 実験履歴

### YYYYMMDD - [実験タイトル]
**目的**: [達成したいこと]
**アプローチ**: [手法・モデル・特徴量など]
**結果**: [OOF/CV スコア]
**次のステップ**: [改善案]

---
