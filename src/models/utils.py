import numpy as np
from typing import Tuple

# タスク名と列順を統一管理
TASK_ORDER: Tuple[str, ...] = ("ctr", "ctcvr", "cvr")
TASK_INDEX = {name: idx for idx, name in enumerate(TASK_ORDER)}


def safe_sigmoid(logits: np.ndarray, clip: float = 500.0) -> np.ndarray:
    """数値安定性を考慮したシグモイド計算。"""
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -clip, clip)))


def add_cvr_labels(y: np.ndarray) -> np.ndarray:
    """
    [Click, CV] → [CTR, CTCVR, CVR] への変換ユーティリティ。

    - CTR  = Click
    - CTCVR = Click * CV
    - CVR  = CV (Click=1 の場合のみ有効、その他は NaN)
    """
    if y.ndim != 2 or y.shape[1] != 2:
        raise ValueError(f"Input y must have shape (n_samples, 2) for [Click, CV], got {y.shape}")

    n_samples = y.shape[0]
    y_converted = np.zeros((n_samples, 3), dtype=float)
    # CTR = Click
    y_converted[:, TASK_INDEX["ctr"]] = y[:, 0]
    # CTCVR = Click * CV
    y_converted[:, TASK_INDEX["ctcvr"]] = y[:, 0] * y[:, 1]
    # CVR = CV if Click==1 else NaN
    y_converted[:, TASK_INDEX["cvr"]] = np.where(y[:, 0] == 1, y[:, 1], np.nan)
    return y_converted
