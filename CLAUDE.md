# CLAUDE.md

## プロジェクト概要
このリポジトリはMLコンペ「活動センサーログによる動作分類」のデータ分析およびモデル作成を行う

### 目的
与えられたTrainデータを用いてモデルを作成し、Testデータに対して最も精度の良いモデルを作成するこ

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
└── CLAUDE.md              # このファイル
```

## データ形式

### train_master.csv
- **概要**: 訓練データのラベル情報ファイル
- **フォーマット**: `id,class`
- **行数**: 4,523件（ヘッダー除く）
- **クラス**: `running`, `walking`, `idle`, `stairs`
- **例**:
  ```
  id,class
  train_00000,running
  train_00001,running
  train_00003,walking
  ```

### train/train_*.csv
- **概要**: 訓練データのセンサーログ
- **フォーマット**: `accelerometer_X,accelerometer_Y,accelerometer_Z`
- **データ**: 3軸加速度センサーの時系列データ
- **サンプリング**: 0.5秒刻みで30データポイント、計15秒間
- **例**:
  ```
  accelerometer_X,accelerometer_Y,accelerometer_Z
  8.695741,-1.149217,-6.325481
  -2.485181,6.770803,-1.656788
  ```

### test/test_*.csv
- **概要**: テストデータのセンサーログ
- **フォーマット**: trainデータと同様の3軸加速度センサーデータ
- **サンプリング**: 0.5秒刻みで30データポイント、計15秒間
- **目的**: 学習済みモデルによる動作分類の対象

### sample_submit.csv
- **概要**: 提出ファイルの形式例
- **フォーマット**: `test_id,predicted_class`
- **例**:
  ```
  test_00000,stairs
  test_00001,stairs
  test_00002,walking
  ```

## 実験履歴

### [日付] - [実験タイトル]

**目的**: [このセッションで達成したいこと]

**主な内容**:
- [ポイント1]
- [ポイント2]
- [ポイント3]

**精度**:
- [生成されたファイルやコード]
- [決定事項]

**次のステップ**:
- [ ] [タスク1]
- [ ] [タスク2]

**メモ**:
[重要な気づきや注意点]

---
