# Criteo CTR/CVR/CTCVR モデル比較実験

このスクリプトは CriteoDataset（実データ）を優先的に使用し、未入手の場合は SyntheticCriteoDataset（合成データ）にフォールバックして、CTR/CTCVR/CVR を同一条件で比較します。出力は標準出力と CSV（デフォルト: reports/tables/criteo_experiment.csv）。

## 実験フロー
1. データ読み込み: `CriteoDataset(sample_size)` を試行し、失敗時に `SyntheticCriteoDataset(sample_size)` へ切替。
2. データ分割: `train/val/test` を stratify=y_click で分割（test_size + val_size をまとめて分割後、相対比で val/test を再分割）。
3. モデル学習:
   - ESMM: 共有埋め込み + 2塔 MLP、epochs=5, batch_size=256。
   - MTGBDT: n_estimators=20, max_depth=3, learning_rate=0.1, n_tasks=3, weighting_strategy="mtgbm"。
   - STGBDTBaseline: 3段階学習（CTR→CTCVR→条件付きCVR）、n_estimators=10, max_depth=2。
4. 予測・評価: `evaluate_predictions` で CTR/CTCVR/CVR の AUC と LogLoss を算出。CVRは click=1 のサンプルに限定し、クラスに変動がなければ NaN。
5. 出力: 標準出力で指標を表示し、CSV に書き出す。

## 指標の定義
- CTR: y[:,0]、予測 pCTR = y_pred[:,0]
- CTCVR: click * conversion、予測 pCTCVR = y_pred[:,1]
- CVR: conversion | click=1、予測 pCVR = y_pred[:,2]
- 評価: AUC, LogLoss（CTR/CTCVR/CVR）

## 実行例
```
python scripts/run_criteo_experiment.py \
  --sample_size 5000 \
  --test_size 0.2 \
  --val_size 0.1 \
  --seed 42 \
  --output reports/tables/criteo_experiment.csv
```

## 期待される用途
- 手法間の相対性能比較（小規模セットで高速検証）
- サンプルサイズや木の本数を増やして精度/計算コストのトレードオフを観察
- 実データと合成データで挙動差を確認し、前処理や特徴量設計の効果を検証
