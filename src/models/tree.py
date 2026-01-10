"""Decision tree implementations for MT-GBDT."""
import math
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any

from .tree_utils import _compute_ips_weight
from .utils import safe_sigmoid


class DecisionTreeNode:
    """A single node in the multi-task decision tree."""

    def __init__(self, node_id: int = 0, depth: int = 0):
        self.feature_idx = None
        self.threshold = None
        self.left = None
        self.right = None
        self.is_leaf = False
        self.values = None

        self.node_id = node_id
        self.depth = depth
        self.n_samples = 0
        self.ctr_rate = 0.0
        self.information_gain = 0.0

    def predict(self, X: np.ndarray, n_tasks: int) -> np.ndarray:
        if self.is_leaf:
            return np.tile(self.values, (X.shape[0], 1))

        left_mask = X[:, self.feature_idx] <= self.threshold
        right_mask = ~left_mask

        predictions = np.zeros((X.shape[0], n_tasks))
        if np.any(left_mask):
            left_pred = self.left.predict(X[left_mask], n_tasks)
            predictions[left_mask] = left_pred
        if np.any(right_mask):
            right_pred = self.right.predict(X[right_mask], n_tasks)
            predictions[right_mask] = right_pred
        return predictions


class MultiTaskDecisionTree:
    """Multi-task decision tree that supports various weighting strategies."""

    def __init__(self, 
                 max_depth: int = 3, 
                 min_samples_split: int = 2, 
                 min_samples_leaf: int = 100,
                 n_tasks: int = 2,
                 weighting_strategy: str = "mtgbm",
                 gain_threshold: float = 0.1,
                 track_split_gains: bool = True,
                 verbose_logging: bool = False,
                 is_dynamic_weight: bool = False,
                 gamma: float = 50.0,
                 delta: float = 0.5,
                 random_state: Optional[int] = None,
                 threshold_prop_ctcvr: float = 0.5,
                 threshold_prop_cvr: float = 0.5,
                 kai_alpha: float = 0.05,
                 kai_target_task: int = 1,
                 kai_source_task: int = 0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_tasks = n_tasks
        self.weighting_strategy = weighting_strategy
        self.gain_threshold = gain_threshold
        self.track_split_gains = track_split_gains
        self.verbose_logging = verbose_logging
        self.is_dynamic_weight = is_dynamic_weight
        self.gamma = gamma
        self.delta = delta
        self.root = None
        self.threshold_prop_ctcvr = threshold_prop_ctcvr
        self.threshold_prop_cvr = threshold_prop_cvr
        self.kai_alpha = kai_alpha
        self.kai_target_task = kai_target_task
        self.kai_source_task = kai_source_task

        self.split_gains = []
        self.current_iteration = 0
        self.change_div_nodes = 0
        self.all_div_nodes = 0

        # propose_kai bookkeeping
        self.kai_switches = 0  # times we fell back to source task
        self.kai_split_calls = 0  # number of split-evaluation attempts under propose_kai

        # propose bookkeeping (delta-based fallback)
        self.propose_switches = 0
        self.propose_split_calls = 0

        # per-split logging
        self.split_logs: List[Dict[str, Any]] = []

        self.node_logs = []
        self.node_counter = 0

        self.gradient_weights = None
        if self.weighting_strategy == "adaptive_hybrid":
            self.gradient_weights = np.array([10.0, 10.0, 1.0])

        self.selected_task = None
        if self.weighting_strategy == "mtgbm":
            if random_state is not None:
                np.random.seed(random_state)
            self.selected_task = np.random.choice([0, 1])

        if random_state is not None:
            np.random.seed(random_state)
    
    def fit(self, 
            X: np.ndarray, 
            raw_gradients: np.ndarray, 
            raw_hessians: np.ndarray,
            split_gradients: np.ndarray,
            split_hessians: np.ndarray,
            y_true: Optional[np.ndarray] = None,
            iteration: int = 0,
            current_predictions: Optional[np.ndarray] = None) -> 'MultiTaskDecisionTree':
        self.n_tasks = raw_gradients.shape[1]
        self.raw_gradients = raw_gradients
        self.raw_hessians = raw_hessians
        self.split_gradients = split_gradients
        self.split_hessians = split_hessians
        
        self.root = DecisionTreeNode()
        self._build_tree(
            self.root, X, np.arange(X.shape[0]),
            depth=0, y_true=y_true, current_predictions=current_predictions
        )
        return self
    
    def _compute_ensemble_weights_gradients_hessians(self, 
                                                    gradients: np.ndarray, 
                                                    hessians: np.ndarray,
                                                    task_weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        n_samples, n_tasks = gradients.shape
        mask = None
        if gradients.shape[1] == 3:
            mask = ~np.isnan(gradients)
        
        if self.weighting_strategy == "ctcvr_subctr_de_norm" or self.weighting_strategy == "mtgbm-de-norm":
            normalized_gradients = gradients
            normalized_hessians = hessians
        else:
            normalized_gradients = self._compute_normalization_weights(gradients, target_mean=0.05, target_std=0.01, mask=mask)
            normalized_hessians = self._compute_normalization_weights(hessians, target_mean=1.00, target_std=0.1, mask=mask)
        if task_weights is not None:
            weights = task_weights
        elif self.weighting_strategy == 'ctcvr-subctr' and n_tasks == 3:
            weights = np.array([1.0, 10.0, 0.0])
        elif self.gradient_weights is not None:
            weights = self.gradient_weights
        else:
            weights = np.ones(n_tasks) / n_tasks
        ensemble_gradients = np.sum(normalized_gradients * weights, axis=1)
        ensemble_hessians = np.sum(normalized_hessians * weights, axis=1)
        
        return ensemble_gradients, ensemble_hessians
    
    def _compute_ensemble_weights_gradients_hessians_afterIPS(self, 
                                                            gradients: np.ndarray, 
                                                            hessians: np.ndarray,
                                                            probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_samples, n_tasks = gradients.shape
        mask = None
        if gradients.shape[1] == 3:
            mask = ~np.isnan(gradients)
        
        normalized_gradients = self._compute_normalization_weights(gradients, target_mean=0.05, target_std=0.01, mask=mask)
        normalized_hessians = self._compute_normalization_weights(hessians, target_mean=1.00, target_std=0.1, mask=mask)
        
        if self.n_tasks == 3:
            ctr_pred = probs[:, 0]
            ips_weights = _compute_ips_weight(ctr_pred)
            cvr_mask = ~np.isnan(gradients[:, 2])
            normalized_gradients[cvr_mask, 2] *= ips_weights[cvr_mask]
            normalized_hessians[cvr_mask, 2] *= ips_weights[cvr_mask]
            nan_mask = np.isnan(gradients[:, 2])
            normalized_gradients[nan_mask, 2] = 0.0
            normalized_hessians[nan_mask, 2] = 0.0
        
        if self.gradient_weights is None:
            weights = np.ones(n_tasks) / n_tasks
        else:
            weights = self.gradient_weights
        
        ensemble_gradients = np.sum(normalized_gradients * weights, axis=1)
        ensemble_hessians = np.sum(normalized_hessians * weights, axis=1)
        
        return ensemble_gradients, ensemble_hessians
    
    def _compute_normalization_weights(self, data: np.ndarray, target_mean: float, target_std: float, mask: Optional[np.ndarray] = None) -> np.ndarray:
        n_samples, n_tasks = data.shape
        normalized_data = np.zeros_like(data)
        
        for task_idx in range(n_tasks):
            task_data = data[:, task_idx]
            
            if mask is not None:
                task_mask = mask[:, task_idx]
                valid_data = task_data[task_mask]
                
                if len(valid_data) == 0:
                    normalized_data[:, task_idx] = task_data.copy()
                    continue
                    
                current_mean = np.mean(valid_data)
                current_std = np.std(valid_data)
                
                if current_std == 0:
                    normalized_data[task_mask, task_idx] = target_mean
                    normalized_data[~task_mask, task_idx] = task_data[~task_mask]
                else:
                    normalized_data[task_mask, task_idx] = (valid_data - current_mean) / current_std * target_std + target_mean
                    normalized_data[~task_mask, task_idx] = task_data[~task_mask]
            else:
                current_mean = np.mean(task_data)
                current_std = np.std(task_data)
                
                if current_std == 0:
                    normalized_data[:, task_idx] = np.full_like(task_data, target_mean)
                else:
                    normalized_data[:, task_idx] = (task_data - current_mean) / current_std * target_std + target_mean
        
        return normalized_data
    
    def _build_tree(self, 
                node: DecisionTreeNode, 
                X: np.ndarray, 
                indices: np.ndarray,
                depth: int,
                y_true: Optional[np.ndarray] = None,
                current_predictions: Optional[np.ndarray] = None):
        n_node_samples = len(indices)
        node.n_samples = n_node_samples
        node.depth = depth
        if y_true is not None:
            node.ctr_rate = np.mean(y_true[indices, 0]) if n_node_samples > 0 else 0.0

        is_leaf_node = (
            depth >= self.max_depth or
            n_node_samples < self.min_samples_split or
            (y_true is not None and len(np.unique(y_true[indices, 0])) == 1)
        )

        best_split = self._find_best_split(X, indices, y_true)

        if is_leaf_node or best_split is None:
            node.is_leaf = True
            node.values = self._calculate_leaf_value(indices)
            self._log_node_info(node, is_leaf=True)
            return

        node.feature_idx = best_split['feature_idx']
        node.threshold = best_split['threshold']
        node.information_gain = best_split['gain']

        left_indices = best_split['left_indices']
        right_indices = best_split['right_indices']

        # record split log (per-split diagnostics)
        split_entry = {
            'depth': node.depth,
            'information_gain': node.information_gain,
            'switched': best_split.get('switched', False),
            'delta': self.delta,
            'weighting_strategy': self.weighting_strategy,
            'node_id': node.node_id,
        }
        self.split_logs.append(split_entry)

        self._log_node_info(node, is_leaf=False, weight_switched=best_split.get('switched', False))

        self.node_counter += 1
        node.left = DecisionTreeNode(node_id=self.node_counter, depth=depth + 1)
        self._build_tree(node.left, X, left_indices, depth + 1, y_true, current_predictions)

        self.node_counter += 1
        node.right = DecisionTreeNode(node_id=self.node_counter, depth=depth + 1)
        self._build_tree(node.right, X, right_indices, depth + 1, y_true, current_predictions)
    
    def _calculate_leaf_value(self, indices: np.ndarray) -> np.ndarray:
        leaf_values = np.zeros(self.n_tasks)
        gradients_subset = self.raw_gradients[indices]
        hessians_subset = self.raw_hessians[indices]

        if self.weighting_strategy == "mtgbm" and self.selected_task is not None:
            for task_idx in range(self.n_tasks):
                grad = gradients_subset[:, task_idx]
                hess = hessians_subset[:, task_idx]
                if np.any(np.isnan(grad)):
                    valid_mask = ~np.isnan(grad)
                    grad = grad[valid_mask]
                    hess = hess[valid_mask]
                sum_grad = np.sum(grad)
                sum_hess = np.sum(hess)
                if sum_hess == 0:
                    leaf_values[task_idx] = 0.0
                else:
                    leaf_values[task_idx] = -sum_grad / sum_hess
        elif self.weighting_strategy == "adaptive_hybrid" and self.n_tasks == 3:
            grad_ctr = gradients_subset[:, 0]
            hess_ctr = hessians_subset[:, 0]
            ctr_value = -np.sum(grad_ctr) / np.sum(hess_ctr) if np.sum(hess_ctr) != 0 else 0.0

            grad_cvr = gradients_subset[:, 2]
            hess_cvr = hessians_subset[:, 2]
            valid_mask = ~np.isnan(grad_cvr)
            grad_cvr = grad_cvr[valid_mask]
            hess_cvr = hess_cvr[valid_mask]
            cvr_value = -np.sum(grad_cvr) / np.sum(hess_cvr) if np.sum(hess_cvr) != 0 else 0.0

            ct_cvr_value = ctr_value * cvr_value

            leaf_values[0] = ctr_value
            leaf_values[1] = ct_cvr_value
            leaf_values[2] = cvr_value
        else:
            for task_idx in range(self.n_tasks):
                grad = gradients_subset[:, task_idx]
                hess = hessians_subset[:, task_idx]
                if np.any(np.isnan(grad)):
                    valid_mask = ~np.isnan(grad)
                    grad = grad[valid_mask]
                    hess = hess[valid_mask]
                sum_grad = np.sum(grad)
                sum_hess = np.sum(hess)
                if sum_hess == 0:
                    leaf_values[task_idx] = 0.0
                else:
                    leaf_values[task_idx] = -sum_grad / sum_hess
        return leaf_values
    

    def _search_best_split(self, 
                          X: np.ndarray, 
                          ensemble_gradients: np.ndarray, 
                          ensemble_hessians: np.ndarray) -> Tuple[float, int, float, np.ndarray, np.ndarray]:
        n_samples = X.shape[0]
        best_gain = 0.0
        best_feature_idx = None
        best_threshold = None
        best_left_indices = None
        best_right_indices = None
        
        for feature_idx in range(X.shape[1]):
            sorted_indices = np.argsort(X[:, feature_idx])
            sorted_feature = X[sorted_indices, feature_idx]
            sorted_ensemble_gradients = ensemble_gradients[sorted_indices]
            sorted_ensemble_hessians = ensemble_hessians[sorted_indices]
            
            left_sum_gradients = np.cumsum(sorted_ensemble_gradients)
            left_sum_hessians = np.cumsum(sorted_ensemble_hessians)
            
            right_sum_gradients = left_sum_gradients[-1] - left_sum_gradients
            right_sum_hessians = left_sum_hessians[-1] - left_sum_hessians
            
            for i in range(1, n_samples):
                if sorted_feature[i] == sorted_feature[i-1]:
                    continue
                
                if i < self.min_samples_leaf or n_samples - i < self.min_samples_leaf:
                    continue
                
                left_gradient = left_sum_gradients[i-1]
                left_hessian = left_sum_hessians[i-1]
                right_gradient = right_sum_gradients[i-1]
                right_hessian = right_sum_hessians[i-1]
                
                lambda_reg = 0.01
                gamma_reg = 0.0
                
                if left_hessian > 0 and right_hessian > 0 and (left_hessian + right_hessian) > 0:
                    gain = 0.5 * (
                        (left_gradient**2) / (left_hessian + lambda_reg) +
                        (right_gradient**2) / (right_hessian + lambda_reg) -
                        ((left_gradient + right_gradient)**2) / (left_hessian + right_hessian + lambda_reg)
                    )
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_feature_idx = feature_idx
                        best_threshold = (sorted_feature[i-1] + sorted_feature[i]) / 2
                        best_left_indices = sorted_indices[:i]
                        best_right_indices = sorted_indices[i:]
        
        return best_gain, best_feature_idx, best_threshold, best_left_indices, best_right_indices

    def _find_best_split(self, X: np.ndarray, indices: np.ndarray, y_true: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
        n_node_samples = len(indices)
        if n_node_samples < self.min_samples_split:
            return None

        X_node = X[indices]
        gradients_node = self.split_gradients[indices]
        hessians_node = self.split_hessians[indices]

        # Dispatch table keeps weighting_strategy-specific branching in dedicated helpers.
        handlers = {
            'ctcvr_subctr_de_norm_gain': self._find_split_ctcvr_subctr_de_norm_gain,
            'adaptive_hybrid': self._find_split_adaptive_hybrid,
            'ctcvr-subctr': self._find_split_ctcvr_subctr,
            'mtgbm': self._find_split_mtgbm,
            'mtgbm-de-norm': self._find_split_mtgbm,
            'ablation_STGBDT_normalize': self._find_split_ablation_normalize,
            'propose': self._find_split_propose,
            'propose_kai': self._find_split_propose_kai,
        }

        handler = handlers.get(self.weighting_strategy, self._find_split_default)
        return handler(X_node, indices, gradients_node, hessians_node, y_true)

    def _find_split_ctcvr_subctr_de_norm_gain(self, X_node: np.ndarray, indices: np.ndarray, gradients: np.ndarray, hessians: np.ndarray, y_true: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
        self.all_div_nodes += 1
        self.propose_split_calls += 1
        switched = False
        best_gain, best_feature_idx, best_threshold, left_indices_local, right_indices_local = self._search_best_split(
            X_node, gradients[:, 1], hessians[:, 1]
        )

        if best_gain < self.delta:
            best_gain, best_feature_idx, best_threshold, left_indices_local, right_indices_local = self._search_best_split(
                X_node, gradients[:, 0], hessians[:, 0]
            )
            self.change_div_nodes += 1
            self.propose_switches += 1
            switched = True

        if best_gain > self.gain_threshold:
            return {
                'gain': best_gain,
                'feature_idx': best_feature_idx,
                'threshold': best_threshold,
                'left_indices': indices[left_indices_local],
                'right_indices': indices[right_indices_local],
                'switched': switched,
            }
        return None

    def _find_split_adaptive_hybrid(self, X_node: np.ndarray, indices: np.ndarray, gradients: np.ndarray, hessians: np.ndarray, y_true: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
        raw_gradients_node = self.raw_gradients[indices]
        prop_neg_ctcvr = np.mean(raw_gradients_node[:, 1] < 0)
        cvr_gradients = raw_gradients_node[:, 2]
        valid_cvr_mask = ~np.isnan(cvr_gradients)
        prop_neg_cvr = np.mean(cvr_gradients[valid_cvr_mask] < 0) if np.any(valid_cvr_mask) else 0.0

        use_ctcvr = prop_neg_ctcvr >= self.threshold_prop_ctcvr
        use_cvr = prop_neg_cvr >= self.threshold_prop_cvr

        w_ctr = 10.0
        w_ctcvr = 10.0 if use_ctcvr else 0.0
        w_cvr = 1.0 if use_cvr else 0.0
        task_weights = np.array([w_ctr, w_ctcvr, w_cvr])

        ensemble_gradients, ensemble_hessians = self._compute_ensemble_weights_gradients_hessians(
            gradients, hessians, task_weights
        )

        best_gain, best_feature_idx, best_threshold, left_indices_local, right_indices_local = self._search_best_split(
            X_node, ensemble_gradients, ensemble_hessians
        )

        if best_gain > 0:
            return {
                'gain': best_gain,
                'feature_idx': best_feature_idx,
                'threshold': best_threshold,
                'left_indices': indices[left_indices_local],
                'right_indices': indices[right_indices_local],
                'switched': False,
            }
        return None

    def _find_split_ctcvr_subctr(self, X_node: np.ndarray, indices: np.ndarray, gradients: np.ndarray, hessians: np.ndarray, y_true: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
        raw_gradients_node = self.raw_gradients[indices]
        prop_neg_ctcvr = np.mean(raw_gradients_node[:, 1] < 0)
        use_ctcvr = prop_neg_ctcvr >= self.threshold_prop_ctcvr
        task_weights = np.array([1.0, 10.0 if use_ctcvr else 0.0, 0.0])

        ensemble_gradients, ensemble_hessians = self._compute_ensemble_weights_gradients_hessians(
            gradients, hessians, task_weights
        )

        best_gain, best_feature_idx, best_threshold, left_indices_local, right_indices_local = self._search_best_split(
            X_node, ensemble_gradients, ensemble_hessians
        )

        if best_gain > 0:
            return {
                'gain': best_gain,
                'feature_idx': best_feature_idx,
                'threshold': best_threshold,
                'left_indices': indices[left_indices_local],
                'right_indices': indices[right_indices_local],
                'switched': False,
            }
        return None

    def _find_split_mtgbm(self, X_node: np.ndarray, indices: np.ndarray, gradients: np.ndarray, hessians: np.ndarray, y_true: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
        task_weights = None
        if self.n_tasks >= 2:
            task_weights = np.zeros(self.n_tasks)
            task_weights[self.selected_task] = 1.0

        ensemble_gradients, ensemble_hessians = self._compute_ensemble_weights_gradients_hessians(
            gradients, hessians, task_weights
        )

        best_gain, best_feature_idx, best_threshold, left_indices_local, right_indices_local = self._search_best_split(
            X_node, ensemble_gradients, ensemble_hessians
        )

        if best_gain > 0:
            return {
                'gain': best_gain,
                'feature_idx': best_feature_idx,
                'threshold': best_threshold,
                'left_indices': indices[left_indices_local],
                'right_indices': indices[right_indices_local],
                'switched': False,
            }
        return None

    def _find_split_ablation_normalize(self, X_node: np.ndarray, indices: np.ndarray, gradients: np.ndarray, hessians: np.ndarray, y_true: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
        mask = None
        if gradients.shape[1] == 3:
            mask = ~np.isnan(gradients)
        normalized_gradients = self._compute_normalization_weights(gradients, target_mean=0.05, target_std=0.01, mask=mask)
        normalized_hessians = self._compute_normalization_weights(hessians, target_mean=1.00, target_std=0.1, mask=mask)

        best_gain = 0.0
        best_feature_idx = None
        best_threshold = None
        best_left_indices = None
        best_right_indices = None

        for task_idx in range(self.n_tasks):
            task_weights = np.zeros(self.n_tasks)
            task_weights[task_idx] = 1.0
            ensemble_gradients = np.sum(normalized_gradients * task_weights, axis=1)
            ensemble_hessians = np.sum(normalized_hessians * task_weights, axis=1)
            gain, feature_idx, threshold, left_indices_local, right_indices_local = self._search_best_split(
                X_node, ensemble_gradients, ensemble_hessians
            )
            if gain > best_gain:
                best_gain = gain
                best_feature_idx = feature_idx
                best_threshold = threshold
                best_left_indices = left_indices_local
                best_right_indices = right_indices_local

        if best_gain > 0:
            return {
                'gain': best_gain,
                'feature_idx': best_feature_idx,
                'threshold': best_threshold,
                'left_indices': indices[best_left_indices],
                'right_indices': indices[best_right_indices],
                'switched': False,
            }
        return None

    def _find_split_propose(self, X_node: np.ndarray, indices: np.ndarray, gradients: np.ndarray, hessians: np.ndarray, y_true: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
        """Delta-gated task switch: try target task, fallback to source if gain is weak."""
        self.propose_split_calls += 1
        target_idx = self.kai_target_task
        source_idx = self.kai_source_task

        def _select_column(arr: np.ndarray, col_idx: int) -> np.ndarray:
            col = arr[:, col_idx]
            if np.any(np.isnan(col)):
                col = np.nan_to_num(col, nan=0.0)
            return col

        target_grad = _select_column(gradients, target_idx)
        target_hess = _select_column(hessians, target_idx)

        best_gain, best_feature_idx, best_threshold, left_indices_local, right_indices_local = self._search_best_split(
            X_node, target_grad, target_hess
        )

        if best_gain > self.delta and best_feature_idx is not None:
            return {
                'gain': best_gain,
                'feature_idx': best_feature_idx,
                'threshold': best_threshold,
                'left_indices': indices[left_indices_local],
                'right_indices': indices[right_indices_local],
                'switched': False,
            }

        self.propose_switches += 1
        return self._search_with_source(indices, X_node, gradients, hessians, source_idx, switched=True)

    def _find_split_default(self, X_node: np.ndarray, indices: np.ndarray, gradients: np.ndarray, hessians: np.ndarray, y_true: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
        if self.n_tasks == 1:
            ensemble_gradients = self.raw_gradients[indices].flatten()
            ensemble_hessians = self.raw_hessians[indices].flatten()
        else:
            ensemble_gradients, ensemble_hessians = self._compute_ensemble_weights_gradients_hessians(
                gradients, hessians, None
            )

        best_gain, best_feature_idx, best_threshold, left_indices_local, right_indices_local = self._search_best_split(
            X_node, ensemble_gradients, ensemble_hessians
        )

        if best_gain > 0:
            return {
                'gain': best_gain,
                'feature_idx': best_feature_idx,
                'threshold': best_threshold,
                'left_indices': indices[left_indices_local],
                'right_indices': indices[right_indices_local],
                'switched': False,
            }
        return None

    def _find_split_propose_kai(self, X_node: np.ndarray, indices: np.ndarray, gradients: np.ndarray, hessians: np.ndarray, y_true: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
        """Chi-square based dynamic task switching: target (ctcvr) first, otherwise source (ctr)."""
        self.kai_split_calls += 1
        target_idx = self.kai_target_task
        source_idx = self.kai_source_task

        def _select_column(arr: np.ndarray, col_idx: int) -> np.ndarray:
            col = arr[:, col_idx]
            if np.any(np.isnan(col)):
                col = np.nan_to_num(col, nan=0.0)
            return col

        # Step 1: search with target task only
        target_grad = _select_column(gradients, target_idx)
        target_hess = _select_column(hessians, target_idx)

        best_gain, best_feature_idx, best_threshold, left_indices_local, right_indices_local = self._search_best_split(
            X_node, target_grad, target_hess
        )

        if best_gain <= 0 or best_feature_idx is None or y_true is None:
            # If no viable split or labels unavailable, fallback to source search directly
            self.kai_switches += 1
            return self._search_with_source(indices, X_node, gradients, hessians, source_idx)

        # Step 2: chi-square test on the target labels for this split
        left_idx = indices[left_indices_local]
        right_idx = indices[right_indices_local]
        y_target = y_true[:, target_idx]

        # Use counts; if available, weight by raw hessian to reflect sample weights
        weight_all = self.raw_hessians[:, target_idx] if hasattr(self, 'raw_hessians') else None

        def _weighted_count(mask: np.ndarray, labels: np.ndarray, weights: Optional[np.ndarray]) -> Tuple[float, float]:
            pos_mask = labels == 1
            if weights is None:
                pos = float(np.sum(mask & pos_mask))
                neg = float(np.sum(mask & (~pos_mask)))
            else:
                pos = float(np.sum(weights[mask & pos_mask]))
                neg = float(np.sum(weights[mask & (~pos_mask)]))
            return pos, neg

        left_mask = np.zeros_like(y_target, dtype=bool)
        left_mask[left_idx] = True
        right_mask = ~left_mask

        left_pos, left_neg = _weighted_count(left_mask, y_target, weight_all)
        right_pos, right_neg = _weighted_count(right_mask, y_target, weight_all)

        total_pos = left_pos + right_pos
        total_neg = left_neg + right_neg
        total = total_pos + total_neg

        def _chi2_pvalue(obs_pos: float, obs_neg: float, total_pos: float, total_neg: float, group_total: float) -> Tuple[float, float]:
            if total <= 0 or group_total <= 0:
                return 0.0, 1.0
            exp_pos = group_total * (total_pos / total)
            exp_neg = group_total * (total_neg / total)
            if exp_pos <= 1e-12 or exp_neg <= 1e-12:
                return 0.0, 1.0
            chi2 = ((obs_pos - exp_pos) ** 2) / exp_pos + ((obs_neg - exp_neg) ** 2) / exp_neg
            p_value = math.erfc(math.sqrt(chi2 / 2.0))
            return chi2, p_value

        chi2_L, p_L = _chi2_pvalue(left_pos, left_neg, total_pos, total_neg, left_pos + left_neg)
        chi2_R, _ = _chi2_pvalue(right_pos, right_neg, total_pos, total_neg, right_pos + right_neg)
        chi2 = chi2_L + chi2_R
        p_value = math.erfc(math.sqrt(chi2 / 2.0)) if total > 0 else 1.0

        if p_value < self.kai_alpha:
            return {
                'gain': best_gain,
                'feature_idx': best_feature_idx,
                'threshold': best_threshold,
                'left_indices': left_idx,
                'right_indices': right_idx,
                'switched': False,
            }

        # Step 3: fallback to source task search
        self.kai_switches += 1
        return self._search_with_source(indices, X_node, gradients, hessians, source_idx, switched=True)

    def _search_with_source(self, indices: np.ndarray, X_node: np.ndarray, gradients: np.ndarray, hessians: np.ndarray, source_idx: int, switched: bool = False) -> Optional[Dict[str, Any]]:
        source_grad = gradients[:, source_idx]
        source_hess = hessians[:, source_idx]
        if np.any(np.isnan(source_grad)):
            source_grad = np.nan_to_num(source_grad, nan=0.0)
        if np.any(np.isnan(source_hess)):
            source_hess = np.nan_to_num(source_hess, nan=0.0)

        best_gain, best_feature_idx, best_threshold, left_indices_local, right_indices_local = self._search_best_split(
            X_node, source_grad, source_hess
        )

        if best_gain > 0 and best_feature_idx is not None:
            return {
                'gain': best_gain,
                'feature_idx': best_feature_idx,
                'threshold': best_threshold,
                'left_indices': indices[left_indices_local],
                'right_indices': indices[right_indices_local],
                'switched': switched,
            }
        return None
    
    def _log_node_info(self, node: DecisionTreeNode, is_leaf: bool = False, weight_switched: bool = False):
        log_entry = {
            'node_id': node.node_id,
            'depth': node.depth,
            'n_samples': node.n_samples,
            'ctr_rate': node.ctr_rate,
            'is_leaf': is_leaf,
            'feature_idx': node.feature_idx if not is_leaf else None,
            'threshold': node.threshold if not is_leaf else None,
            'information_gain': node.information_gain if not is_leaf else 0.0,
            'leaf_values': node.values.tolist() if hasattr(node.values, 'tolist') else node.values,
            'weight_switched': weight_switched,
            'delta': self.delta,
        }
        
        self.node_logs.append(log_entry)
        
        if self.verbose_logging:
            if is_leaf:
                print(f"Leaf Node {node.node_id}: depth={node.depth}, samples={node.n_samples}, CVR={node.ctr_rate:.4f}")
            else:
                print(f"Split Node {node.node_id}: depth={node.depth}, samples={node.n_samples}, CVR={node.ctr_rate:.4f}, gain={node.information_gain:.4f}, feature={node.feature_idx}, threshold={node.threshold:.4f}, weight_switched={weight_switched}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise ValueError("Model has not been trained yet")
        n_tasks = self.n_tasks
        return self.root.predict(X, n_tasks)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = self.predict(X)
        y_proba = safe_sigmoid(logits)
        if (hasattr(self, 'weighting_strategy') and self.weighting_strategy == "mtgbm-ctr-cvr" and
            self.n_tasks == 3):
            y_proba[:, 1] = y_proba[:, 0] * y_proba[:, 2]
        return y_proba
