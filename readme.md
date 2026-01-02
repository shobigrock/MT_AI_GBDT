# MT_AI_GBDT プロジェクトガイド

CTR/CVR/CTCVR 予測向けに GBDT を改良する研究・実験用のリポジトリです。データ前処理から特徴量設計、学習・評価、実験管理までを一貫して進められる構成を用意しています。

## ディレクトリ概要

| パス | 用途 |
| --- | --- |
| [data/raw](data/raw) | 生データの格納。外部入手データをそのまま置く領域 |
| [data/interim](data/interim) | 中間生成物（クリーニング・結合済みなど） |
| [data/processed](data/processed) | 学習・評価で直接使う確定データ |
| [data/features](data/features) | 事前計算した特徴量キャッシュ（例: 集約統計、embedding など） |
| [notebooks/prototype](notebooks/prototype) | アイデア検証用ノート（小規模データで素早く試す） |
| [notebooks/analysis](notebooks/analysis) | 解析・可視化・結果考察ノート |
| [src/data](src/data) | 取り込み・前処理ロジック（スキーマ検証、フィルタリング、欠損処理など） |
| [src/features](src/features) | 特徴量生成・変換（頻度統計、クロス特徴、時系列ウィンドウなど） |
| [src/models](src/models) | 改良 GBDT 実装と学習・推論ラッパー |
| [src/pipelines](src/pipelines) | end-to-end パイプライン定義（データ→特徴→学習→評価） |
| [src/metrics](src/metrics) | CTR/CVR/CTCVR 向け指標（AUC、LogLoss、Calibration、PR-AUC など） |
| [src/utils](src/utils) | 共通ユーティリティ（ロギング、シード固定、パス管理） |
| [configs/data](configs/data) | データ設定（入力パス、分割比、フィルタ条件） |
| [configs/model](configs/model) | モデル設定（ハイパーパラメータ、学習率スケジュール、正則化） |
| [configs/experiment](configs/experiment) | 実験設定（使用データ・特徴・モデルの組み合わせ、乱数シード） |
| [experiments/runs](experiments/runs) | 実行ごとの成果物（設定スナップショット、ログ、指標、図表） |
| [experiments/ablation](experiments/ablation) | アブレーション比較結果のまとめ |
| [scripts](scripts) | CLI スクリプト置き場（例: `prepare_data`, `train`, `evaluate`） |
| [models/checkpoints](models/checkpoints) | 学習済みモデルのチェックポイント |
| [models/artifacts](models/artifacts) | 変換済み特徴量辞書や前処理パイプラインのシリアライズ |
| [reports/figures](reports/figures) | 可視化画像（学習曲線、重要度、Calibration プロット等） |
| [reports/tables](reports/tables) | 指標テーブル、比較表 |
| [logs/training](logs/training) | 学習時ログ（ハイパーパラメータ、損失推移） |
| [logs/evaluation](logs/evaluation) | 評価時ログ（指標、エラー解析出力） |
| [references](references) | 論文・メモ・関連資料 |
| [tests](tests) | ユニットテスト・回帰テスト |

## 典型ワークフロー

1. データ投入: 生データを data/raw に配置し、前処理スクリプト（例: scripts/prepare_data.py）で data/interim → data/processed を生成。
2. 特徴量設計: notebooks/prototype で試作し、確定した処理を src/features に移植。計算結果を data/features にキャッシュ。
3. 設定準備: configs/data・configs/model・configs/experiment に YAML/JSON で設定を書く。実験ごとにバージョン管理。
4. 学習実行: scripts/train.py から src/pipelines のパイプラインを呼び出し、models/checkpoints と logs/training に保存。
5. 評価・可視化: scripts/evaluate.py で指標を算出し、logs/evaluation と reports/figures/reports/tables にまとめる。
6. 実験整理: 実行時の設定ファイルと結果を experiments/runs/日付_識別子 にコピーし、ablation 比較は experiments/ablation に集約。

## 実験運用のヒント

- 設定スナップショット: 実行前に使用した configs を experiments/runs/<run_id>/configs に保存すると再現しやすくなります。
- シード固定: src/utils にシード初期化関数を置き、学習・評価で毎回呼び出してください。
- ロギング: 学習/評価スクリプトは logs 配下に出力しつつ、標準出力も tee すると便利です。
- メトリクス: CTR/CVR/CTCVR 用に AUC・LogLoss・ECE/Calibration・PR-AUC を最低限出すようにします。

## 今後置くと便利なテンプレ

- scripts/prepare_data.py: raw → processed への前処理 CLI
- scripts/train.py: パイプライン実行、モデル保存、ログ出力
- scripts/evaluate.py: 評価指標計算と可視化生成
- configs/data/*.yml, configs/model/*.yml, configs/experiment/*.yml: デフォルト設定サンプル
- tests/: 前処理・特徴量関数・メトリクス計算のユニットテスト

## 次の一手（例）

- デフォルト設定の雛形 (data/model/experiment) を作成
- 前処理と学習の最小パイプラインを scripts から呼べる状態にする
- プロトタイプノートで特徴量候補を洗い出し、重要度やカバレッジを確認
