"""Tree-related helper functions shared across MT-GBDT components."""
import numpy as np


def _compute_ips_weight(ctr_pred: np.ndarray, epsilon: float = 1e-3) -> np.ndarray:
    """Compute inverse propensity scores for CVR task with floor epsilon."""
    return 1.0 / np.maximum(ctr_pred, epsilon)


def _convert_to_ordinal_labels(y_click_cv: np.ndarray) -> np.ndarray:
    """Convert two-column (click, cv) labels to ordinal 0/1/2 encoding."""
    y_click = y_click_cv[:, 0]
    y_cv = y_click_cv[:, 1]
    ordinal = np.zeros_like(y_click, dtype=int)
    ordinal[(y_click == 1) & (y_cv == 0)] = 1
    ordinal[(y_click == 1) & (y_cv == 1)] = 2
    return ordinal.reshape(-1, 1)
