import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import train_test_split

# プロジェクトルートをパスに追加
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.real_data import CriteoDataset, SyntheticCriteoDataset
from src.models.esmm import ESMM
from src.models.gbdt_proto import MTGBDT
from src.models.stgbdt import STGBDTBaseline
from src.models.utils import add_cvr_labels


def load_criteo(sample_size: int, random_state: int) -> Tuple[np.ndarray, np.ndarray]:
    """CriteoDataset を試み、ファイル未存在時は SyntheticCriteoDataset にフォールバック。"""
    dataset = CriteoDataset(sample_size=sample_size)
    try:
        X, y = dataset.get_data(random_state=random_state)
        return X, y
    except Exception:
        synth = SyntheticCriteoDataset(sample_size=sample_size)
        X, y = synth.load_data(random_state=random_state)
        return X, y


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """CTR/CTCVR/CVR の AUC と LogLoss を計算する。"""
    eps = 1e-8
    y_click = y_true[:, 0]
    y_conv = y_true[:, 1]
    y_ctcvr = y_click * y_conv

    pred_ctr = np.clip(y_pred[:, 0], eps, 1 - eps)
    pred_ctcvr = np.clip(y_pred[:, 1], eps, 1 - eps)
    pred_cvr = np.clip(y_pred[:, 2], eps, 1 - eps)

    metrics = {}

    # CTR
    metrics["auc_ctr"] = roc_auc_score(y_click, pred_ctr)
    metrics["logloss_ctr"] = log_loss(y_click, pred_ctr)

    # CTCVR
    metrics["auc_ctcvr"] = roc_auc_score(y_ctcvr, pred_ctcvr)
    metrics["logloss_ctcvr"] = log_loss(y_ctcvr, pred_ctcvr)

    # CVR: click==1 に限定
    click_mask = y_click == 1
    if np.sum(click_mask) > 1 and np.unique(y_conv[click_mask]).size > 1:
        metrics["auc_cvr"] = roc_auc_score(y_conv[click_mask], pred_cvr[click_mask])
        metrics["logloss_cvr"] = log_loss(y_conv[click_mask], pred_cvr[click_mask])
    else:
        metrics["auc_cvr"] = np.nan
        metrics["logloss_cvr"] = np.nan

    return metrics


def run_experiment(sample_size: int, random_state: int, test_size: float, val_size: float) -> List[Dict[str, float]]:
    X, y = load_criteo(sample_size=sample_size, random_state=random_state)

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=test_size + val_size, random_state=random_state, stratify=y[:, 0]
    )
    rel_val = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=1 - rel_val, random_state=random_state, stratify=y_tmp[:, 0]
    )

    results = []

    # ESMM
    esmm = ESMM(epochs=5, batch_size=256, verbose=0, validation_split=0.0)
    esmm.fit(X_train, y_train)
    esmm_pred = esmm.predict_proba(X_test)
    esmm_metrics = evaluate_predictions(y_test, esmm_pred)
    esmm_metrics.update({"model": "ESMM", "n_params": esmm.model_.count_params() if esmm.model_ else 0})
    results.append(esmm_metrics)

    # MTGBDT (3タスクモードにするため n_tasks=3)
    mtgbdt = MTGBDT(n_estimators=20, learning_rate=0.1, max_depth=3, n_tasks=3, loss="logloss", weighting_strategy="mtgbm")
    mtgbdt.fit(X_train, y_train)
    mtgbdt_pred = mtgbdt.predict_proba(X_test)
    mtgbdt_metrics = evaluate_predictions(y_test, mtgbdt_pred)
    mtgbdt_metrics.update({"model": "MTGBDT", "n_estimators": mtgbdt.n_estimators})
    results.append(mtgbdt_metrics)

    # STGBDT Baseline
    stg = STGBDTBaseline(n_estimators=10, learning_rate=0.3, max_depth=2, min_samples_split=20, min_samples_leaf=10)
    stg.fit(X_train, y_train)
    stg_pred = stg.predict_proba(X_test)
    stg_metrics = evaluate_predictions(y_test, stg_pred)
    stg_metrics.update({"model": "STGBDTBaseline", "n_estimators": stg.n_estimators})
    results.append(stg_metrics)

    return results


def main():
    parser = argparse.ArgumentParser(description="Criteo CTR/CVR/CTCVR モデル比較実験")
    parser.add_argument("--sample_size", type=int, default=5000, help="使用するサンプル数")
    parser.add_argument("--test_size", type=float, default=0.2, help="テスト比率")
    parser.add_argument("--val_size", type=float, default=0.1, help="バリデーション比率")
    parser.add_argument("--seed", type=int, default=42, help="乱数シード")
    parser.add_argument("--output", type=str, default="reports/tables/criteo_experiment.csv", help="結果の保存先CSV")
    args = parser.parse_args()

    results = run_experiment(
        sample_size=args.sample_size,
        random_state=args.seed,
        test_size=args.test_size,
        val_size=args.val_size,
    )

    # 結果表示
    headers = ["model", "auc_ctr", "auc_ctcvr", "auc_cvr", "logloss_ctr", "logloss_ctcvr", "logloss_cvr"]
    print("\n=== Experiment Results ===")
    for res in results:
        row = [res.get(h, np.nan) for h in headers]
        print(dict(zip(headers, row)))

    # 保存
    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
