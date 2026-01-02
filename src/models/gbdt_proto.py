import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.model_selection import KFold

from .base import MTGBMBase
from .config_loader import load_model_config
from .tree import DecisionTreeNode, MultiTaskDecisionTree
from .tree_utils import _compute_ips_weight, _convert_to_ordinal_labels
from .utils import add_cvr_labels, safe_sigmoid


class MTGBDT(MTGBMBase):
    """
    汎用マルチタスク勾配ブースティング決定木(MT-GBDT)のスクラッチ実装

    シングルタスク(n_tasks=1)とマルチタスク(n_tasks>=2)の両方に対応。
    シングルタスクの場合は従来のGBDTと同等の動作をし、
    マルチタスクの場合は各タスク間の相関を活用した学習を行う。

    Attributes:
    -----------
    n_estimators : int
        ブースティング反復回数(木の数)
    learning_rate : float
        学習率(各木の寄与度)
    max_depth : int
        各木の最大深さ
    min_samples_split : int
        分割に必要な最小サンプル数
    min_samples_leaf : int
        リーフノードに必要な最小サンプル数
    subsample : float
        各木を構築する際のサンプリング率
    colsample_bytree : float
        各木を構築する際の特徴サンプリング率
    n_tasks : int
        タスク数
    trees_ : list of MultiTaskDecisionTree
        学習済みの木のリスト
    initial_predictions_ : array-like, shape=(n_tasks,)
        初期予測値
    """

    def __init__(
        self,
        n_estimators: Optional[int] = None,
        learning_rate: Optional[float] = None,
        max_depth: Optional[int] = None,
        min_samples_split: Optional[int] = None,
        min_samples_leaf: Optional[int] = None,
        n_tasks: Optional[int] = None,
        subsample: Optional[float] = None,
        colsample_bytree: Optional[float] = None,
        gradient_weights: Optional[np.ndarray] = None,
        normalize_gradients: Optional[bool] = None,
        loss: Optional[str] = None,
        weighting_strategy: Optional[str] = None,
        gain_threshold: Optional[float] = None,
        track_split_gains: Optional[bool] = None,
        is_dynamic_weight: Optional[bool] = None,
        gamma: Optional[float] = None,
        delta: Optional[float] = None,
        verbose_logging: Optional[bool] = None,
        random_state: Optional[int] = None,
        n_folds_oof: Optional[int] = None,
        threshold_prop_ctcvr: Optional[float] = None,
        threshold_prop_cvr: Optional[float] = None,
        config: Optional[Dict[str, Any]] = None,
        config_mode: Optional[str] = None,
        config_dir: Optional[Union[str, Path]] = None,
        use_config: bool = True,
    ):
        defaults: Dict[str, Any] = {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "n_tasks": 2,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "gradient_weights": None,
            "normalize_gradients": False,
            "loss": "logloss",
            "weighting_strategy": "mtgbm",
            "gain_threshold": 0.1,
            "track_split_gains": True,
            "is_dynamic_weight": False,
            "gamma": 50.0,
            "delta": 0.5,
            "verbose_logging": False,
            "random_state": None,
            "n_folds_oof": 5,
            "threshold_prop_ctcvr": 0.5,
            "threshold_prop_cvr": 0.5,
        }

        config_params: Dict[str, Any] = {}
        if use_config:
            config_params = load_model_config("mtgbdt", mode=config_mode, config_dir=config_dir)
        if config:
            config_params.update(config)

        resolved = {**defaults, **config_params}
        explicit_params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "n_tasks": n_tasks,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "gradient_weights": gradient_weights,
            "normalize_gradients": normalize_gradients,
            "loss": loss,
            "weighting_strategy": weighting_strategy,
            "gain_threshold": gain_threshold,
            "track_split_gains": track_split_gains,
            "is_dynamic_weight": is_dynamic_weight,
            "gamma": gamma,
            "delta": delta,
            "verbose_logging": verbose_logging,
            "random_state": random_state,
            "n_folds_oof": n_folds_oof,
            "threshold_prop_ctcvr": threshold_prop_ctcvr,
            "threshold_prop_cvr": threshold_prop_cvr,
        }

        for key, value in explicit_params.items():
            if value is not None:
                resolved[key] = value

        resolved_gradient_weights = resolved.get("gradient_weights")
        if resolved_gradient_weights is not None and not isinstance(resolved_gradient_weights, np.ndarray):
            resolved_gradient_weights = np.asarray(resolved_gradient_weights, dtype=float)

        super().__init__(
            n_estimators=resolved["n_estimators"],
            learning_rate=resolved["learning_rate"],
            max_depth=resolved["max_depth"],
            random_state=resolved["random_state"],
        )
        """
        マルチタスク勾配ブースティング決定木(MT-GBDT)の初期化
        Parameters:
        -----------
        n_estimators : int
            ブースティング反復回数(木の数)
        learning_rate : float
            学習率(各木の寄与度)
        max_depth : int
            各木の最大深さ
        min_samples_split : int
            分割に必要な最小サンプル数
        min_samples_leaf : int
            リーフノードに必要な最小サンプル数
        n_tasks : int
            タスク数(2以上でマルチタスク)
        subsample : float
            各木を構築する際のサンプリング率(0.0 < subsample <= 1.0)
        colsample_bytree : float
            各木を構築する際の特徴サンプリング率(0.0 < colsample_bytree <= 1.0)
        gradient_weights : np.ndarray, optional
            各タスクの勾配の重み(Noneの場合は均等重み)
        normalize_gradients : bool
            勾配を正規化するかどうか(Trueの場合、勾配を平均0、標準偏差1に正規化)
        loss : str
            損失関数("logloss" または "mse")
        weighting_strategy : str
            勾配の重み付け戦略("mtgbm", "adaptive_hybrid", "mtgbm-ctr-cvr" など)
        gain_threshold : float
            分割の情報利得の閾値(この値以下の分割は行わない)
        track_split_gains : bool
            分割ごとの情報利得を追跡するかどうか(Trueの場合、各分割の利得を記録)
        is_dynamic_weight : bool
            動的重み付けを使用するかどうか(Trueの場合、勾配の重みを動的に調整)
        gamma : float
            動的重み付けのパラメータ(デフォルトは50.0)
        delta : float
            動的重み付けのパラメータ(デフォルトは0.5)
        verbose_logging : bool
            ログ出力を詳細にするかどうか(Trueの場合、各ノードの詳細を出力)
        random_state : int, optional
            乱数シード(再現性のため)
        n_folds_oof : int
            OOF予測のためのフォールド数(adaptive_hybrid戦略用)
        threshold_prop_ctcvr : float
            CTCVRタスクの負の勾配割合の閾値(adaptive_hybrid戦略用)
        threshold_prop_cvr : float
            CVRタスクの負の勾配割合の閾値(adaptive_hybrid戦略用)
        """
        self.min_samples_split = resolved["min_samples_split"]
        self.min_samples_leaf = resolved["min_samples_leaf"]
        self.n_tasks = resolved["n_tasks"]
        self.subsample = resolved["subsample"]
        self.colsample_bytree = resolved["colsample_bytree"]
        self.gradient_weights = resolved_gradient_weights
        self.normalize_gradients = resolved["normalize_gradients"]
        self.loss = resolved["loss"]
        self.weighting_strategy = resolved["weighting_strategy"]
        self.gain_threshold = resolved["gain_threshold"]
        self.track_split_gains = resolved["track_split_gains"]
        self.is_dynamic_weight = resolved["is_dynamic_weight"]
        self.gamma = resolved["gamma"]
        self.delta = resolved["delta"]
        self.verbose_logging = resolved["verbose_logging"]

        self.n_folds_oof = resolved["n_folds_oof"]
        self.threshold_prop_ctcvr = resolved["threshold_prop_ctcvr"]
        self.threshold_prop_cvr = resolved["threshold_prop_cvr"]
        
        self.trees_ = []
        self.initial_predictions_ = None
        self.eval_results_ = {}
        self.best_iteration_ = 0
        self.best_score_ = float('inf')
        self.oof_ctr_preds_ = None
    
    def _compute_initial_predictions(self, y_multi: np.ndarray) -> np.ndarray:
        """
        初期予測値を計算(ログオッズ空間で計算、3タスク対応でCVRのNaN値を適切に処理)
        """
        if self.n_tasks == 3:
            initial_preds = np.zeros(self.n_tasks)

            ctr_prob = np.clip(np.mean(y_multi[:, 0]), 1e-7, 1-1e-7)
            initial_preds[0] = np.log(ctr_prob / (1 - ctr_prob))

            ctcvr_prob = np.clip(np.mean(y_multi[:, 1]), 1e-7, 1-1e-7)
            initial_preds[1] = np.log(ctcvr_prob / (1 - ctcvr_prob))

            cvr_mask = ~np.isnan(y_multi[:, 2])
            if np.sum(cvr_mask) > 0:
                cvr_prob = np.clip(np.mean(y_multi[cvr_mask, 2]), 1e-7, 1-1e-7)
                initial_preds[2] = np.log(cvr_prob / (1 - cvr_prob))
            else:
                initial_preds[2] = 0.0
                
            return initial_preds
        elif self.weighting_strategy == 'ordinal':
            # 2閾値の初期値（全体分布から）
            y_ord = _convert_to_ordinal_labels(y_multi)
            # theta1: P(y>=1), theta2: P(y>=2)
            p1 = np.clip(np.mean(y_ord >= 1), 1e-7, 1-1e-7)
            p2 = np.clip(np.mean(y_ord == 2), 1e-7, 1-1e-7)
            theta1 = np.log(p1 / (1 - p1))
            theta2 = np.log(p2 / (1 - p2))
            return np.array([theta1, theta2])
        else:
            initial_preds = np.zeros(self.n_tasks)
            for task in range(self.n_tasks):
                task_prob = np.clip(np.mean(y_multi[:, task]), 1e-7, 1-1e-7)
                initial_preds[task] = np.log(task_prob / (1 - task_prob))
            return initial_preds
    
    def _compute_raw_gradients_hessians(self, y_multi: np.ndarray, y_pred_logits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        生の勾配とヘシアンを計算する。
        """
        if self.loss == "logloss":
            probs = safe_sigmoid(y_pred_logits)
            gradients = probs - y_multi
            hessians = probs * (1 - probs)
            hessians = np.maximum(hessians, 1e-7)
            if self.n_tasks == 3:
                nan_mask = np.isnan(y_multi[:, 2])
                gradients[nan_mask, 2] = 0.0
                hessians[nan_mask, 2] = 0.0
        elif self.loss == "mse":
            gradients = 2 * (y_pred_logits - y_multi)
            hessians = np.ones_like(gradients) * 2
        else:
            raise ValueError(f"Unsupported loss function: {self.loss}")
        return gradients, hessians

    def _prepare_gradients_for_splitting(self, raw_gradients: np.ndarray, raw_hessians: np.ndarray, current_predictions: np.ndarray, iteration: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        分岐決定用の勾配ヘシアンを準備(IPS補正など)
        """
        split_gradients = raw_gradients.copy()
        split_hessians = raw_hessians.copy()

        if self.n_tasks == 3:
            if self.weighting_strategy == 'adaptive_hybrid':
                if self.oof_ctr_preds_ is None:
                    raise ValueError("OOF CTR predictions are not generated for adaptive_hybrid strategy.")
                ips_weights = _compute_ips_weight(self.oof_ctr_preds_)
            else:
                probs = safe_sigmoid(current_predictions)
                ctr_pred = probs[:, 0]
                ips_weights = _compute_ips_weight(ctr_pred)

            cvr_mask = ~np.isnan(split_gradients[:, 2])
            split_gradients[cvr_mask, 2] *= ips_weights[cvr_mask]
            split_hessians[cvr_mask, 2] *= ips_weights[cvr_mask]

        return split_gradients, split_hessians

    def _calculate_loss(self, y_true, y_pred_logits):
        if self.weighting_strategy == 'ordinal':
            y_ord = _convert_to_ordinal_labels(y_true)
            return self._ordinal_proportional_odds_loss(y_ord, y_pred_logits)
        elif self.loss == "logloss":
            y_pred_proba = safe_sigmoid(y_pred_logits)
            y_pred_proba = np.clip(y_pred_proba, 1e-7, 1 - 1e-7)
            
            if self.n_tasks == 3:
                logloss = - (y_true * np.log(y_pred_proba) + (1 - y_true) * np.log(1 - y_pred_proba))
                return np.nanmean(logloss)
            else:
                return -np.mean(y_true * np.log(y_pred_proba) + (1 - y_true) * np.log(1 - y_pred_proba))
        elif self.loss == "mse":
            return np.mean((y_true - y_pred_logits) ** 2)
        else:
            raise ValueError(f"Unsupported loss: {self.loss}")

    def _generate_oof_ctr_predictions(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Out-of-Fold (OOF) を用いて、学習データに対するCTR予測値を生成します。
        """
        oof_preds = np.zeros(X.shape[0])
        kf = KFold(n_splits=self.n_folds_oof, shuffle=True, random_state=self.random_state)

        oof_model = MTGBDT(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            n_tasks=1,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            loss='logloss',
            weighting_strategy='stgbdt_baseline',
            random_state=self.random_state
        )

        for train_index, val_index in kf.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train_ctr = y[train_index, 0].reshape(-1, 1)

            oof_model.fit(X_train, y_train_ctr)
            oof_preds[val_index] = oof_model.predict_proba(X_val)[:, 0]

        return oof_preds

    def _ordinal_proportional_odds_loss(self, y_ord: np.ndarray, logits: np.ndarray) -> float:
        """
        順序ロジスティック回帰（比例オッズモデル）の損失
        y_ord: (n_samples, 1) 0,1,2
        logits: (n_samples, 2) 2つの閾値
        """
        # logits: shape (n_samples, 2)  (theta_1, theta_2)
        theta1 = logits[:, 0]
        theta2 = logits[:, 1]
        # P(y >= 1) = sigmoid(theta1), P(y >= 2) = sigmoid(theta2)
        p1 = 1 / (1 + np.exp(-theta1))
        p2 = 1 / (1 + np.exp(-theta2))
        # 各クラスの確率
        prob0 = 1 - p1
        prob1 = p1 - p2
        prob2 = p2
        # one-hot
        y0 = (y_ord[:, 0] == 0)
        y1 = (y_ord[:, 0] == 1)
        y2 = (y_ord[:, 0] == 2)
        eps = 1e-9
        loss = -np.mean(y0 * np.log(prob0 + eps) + y1 * np.log(prob1 + eps) + y2 * np.log(prob2 + eps))
        return loss

    def _ordinal_proportional_odds_grad_hess(self, y_ord: np.ndarray, logits: np.ndarray):
        """
        順序ロジスティック回帰の勾配ヘッセ行列
        y_ord: (n_samples, 1)
        logits: (n_samples, 2)
        戻り値: grad, hess (shape: (n_samples, 2))
        """
        theta1 = logits[:, 0]
        theta2 = logits[:, 1]
        p1 = 1 / (1 + np.exp(-theta1))
        p2 = 1 / (1 + np.exp(-theta2))
        y0 = (y_ord[:, 0] == 0)
        y1 = (y_ord[:, 0] == 1)
        y2 = (y_ord[:, 0] == 2)
        grad1 = -y0 * p1 + y1 * (1 - p1) - y1 * p2 + y2 * (0)
        grad2 = -y1 * p2 + y2 * (1 - p2)
        grad = np.stack([grad1, grad2], axis=1)
        # ヘッセ行列（対角のみ）
        hess1 = y0 * p1 * (1 - p1) + y1 * p1 * (1 - p1)
        hess2 = y1 * p2 * (1 - p2) + y2 * p2 * (1 - p2)
        hess = np.stack([hess1, hess2], axis=1)
        hess = np.maximum(hess, 1e-7)
        return grad, hess

    def fit(self, X: np.ndarray, y_multi: np.ndarray, **kwargs) -> 'MTGBDT':
        """
        シングル/マルチタスクデータでモデルを学習
        """
        X, y_multi = self._validate_input(X, y_multi)

        self.trees_ = []
        self.eval_results_ = {}
        self.best_iteration_ = 0
        self.best_score_ = float('inf')
        self.oof_ctr_preds_ = None

        if self.weighting_strategy == 'adaptive_hybrid':
            if self.n_tasks != 3:
                raise ValueError("adaptive_hybrid strategy is only supported for 3 tasks.")
            print("Generating OOF CTR predictions for adaptive_hybrid strategy...")
            self.oof_ctr_preds_ = self._generate_oof_ctr_predictions(X, y_multi)
            print("OOF CTR predictions generated.")

        self.initial_predictions_ = self._compute_initial_predictions(y_multi)
        current_predictions = np.tile(self.initial_predictions_, (X.shape[0], 1))
        
        start_time = time.time()
        
        for i in range(self.n_estimators):
            raw_gradients, raw_hessians = self._compute_raw_gradients_hessians(y_multi, current_predictions)
            split_gradients, split_hessians = self._prepare_gradients_for_splitting(raw_gradients, raw_hessians, current_predictions, i)

            if self.subsample < 1.0:
                sample_indices = np.random.choice(X.shape[0], size=int(X.shape[0] * self.subsample), replace=False)
            else:
                sample_indices = np.arange(X.shape[0])

            X_sampled = X[sample_indices]
            
            feature_indices = None
            if self.colsample_bytree < 1.0:
                n_features = X.shape[1]
                feature_indices = np.random.choice(n_features, size=int(n_features * self.colsample_bytree), replace=False)
                X_sampled = X_sampled[:, feature_indices]

            tree = MultiTaskDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                n_tasks=self.n_tasks,
                weighting_strategy=self.weighting_strategy,
                gain_threshold=self.gain_threshold,
                track_split_gains=self.track_split_gains,
                verbose_logging=self.verbose_logging,
                is_dynamic_weight=self.is_dynamic_weight,
                gamma=self.gamma,
                delta=self.delta,
                random_state=self.random_state,
                threshold_prop_ctcvr=self.threshold_prop_ctcvr,
                threshold_prop_cvr=self.threshold_prop_cvr
            )

            tree.fit(
                X_sampled, 
                raw_gradients[sample_indices], 
                raw_hessians[sample_indices],
                split_gradients[sample_indices],
                split_hessians[sample_indices],
                y_true=y_multi[sample_indices],
                current_predictions=current_predictions[sample_indices]
            )
            
            self.trees_.append({'tree': tree, 'feature_indices': feature_indices})
            
            update = tree.predict(X_sampled)
            np.add.at(current_predictions, sample_indices, self.learning_rate * update)

            if (i + 1) % 10 == 0 or i == 0 or i == self.n_estimators - 1:
                elapsed_time = time.time() - start_time
                loss_value = self._calculate_loss(y_multi, current_predictions)
                loss_name = "LogLoss" if self.loss == "logloss" else "MSE"
                print(f"Iteration {i+1}/{self.n_estimators}, {loss_name}: {loss_value:.6f}, Time: {elapsed_time:.2f}s")

        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        学習済みモデルで予測
        """
        X, _ = self._validate_input(X)
        
        if not self.trees_:
            raise ValueError("Model has not been trained yet")
        
        y_pred = np.tile(self.initial_predictions_, (X.shape[0], 1))
        
        for tree_info in self.trees_:
            tree = tree_info['tree']
            feature_indices = tree_info['feature_indices']
            
            X_subset = X[:, feature_indices] if feature_indices is not None else X
            tree_prediction = tree.predict(X_subset)
            
            y_pred += self.learning_rate * tree_prediction
        
        return y_pred
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        確率予測を行う
        """
        if self.weighting_strategy == 'ordinal':
            # 2次元logit3タスク確率
            logits = np.tile(self.initial_predictions_, (X.shape[0], 1))
            for tree_info in self.trees_:
                tree = tree_info['tree']
                X_subset = X
                tree_prediction = tree.predict(X_subset)
                logits += self.learning_rate * tree_prediction
            p1 = 1 / (1 + np.exp(-logits[:, 0]))
            p2 = 1 / (1 + np.exp(-logits[:, 1]))
            ctr = p1
            ctcvr = p2
            cvr = np.zeros_like(ctr)
            mask = ctr > 1e-7
            cvr[mask] = ctcvr[mask] / ctr[mask]
            cvr[~mask] = 0.0
            return np.stack([ctr, ctcvr, cvr], axis=1)
        else:
            logits = self.predict(X)
            y_proba = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
        
            if (hasattr(self, 'weighting_strategy') and 
                self.weighting_strategy == "mtgbm-ctr-cvr" and 
                self.n_tasks == 3):
                y_proba[:, 1] = y_proba[:, 0] * y_proba[:, 2]
            
            return y_proba
    
    def get_feature_importance(self) -> np.ndarray:
        """
        特徴量の重要度を計算
        """
        if not self.trees_:
            raise ValueError("Model has not been trained yet")
        
        n_features = max(max(tree_info['feature_indices']) for tree_info in self.trees_ if tree_info['feature_indices'] is not None) + 1
        feature_importance = np.zeros(n_features)
        
        for tree_info in self.trees_:
            tree = tree_info['tree']
            self._count_feature_usage(tree.root, feature_importance)
        
        if np.sum(feature_importance) > 0:
            feature_importance = feature_importance / np.sum(feature_importance)
        
        return feature_importance
    
    def _count_feature_usage(self, node: DecisionTreeNode, feature_importance: np.ndarray) -> None:
        """
        ノードで使用される特徴量をカウント
        """
        if node is None or node.is_leaf:
            return
        
        if node.feature_idx is not None:
            feature_importance[node.feature_idx] += 1
        
        self._count_feature_usage(node.left, feature_importance)
        self._count_feature_usage(node.right, feature_importance)
    
    def get_node_logs(self) -> List[Dict]:
        """
        全ての木のノードログを取得
        """
        all_logs = []
        for i, tree_info in enumerate(self.trees_):
            tree = tree_info['tree']
            for log in tree.node_logs:
                log_copy = log.copy()
                log_copy['tree_index'] = i
                all_logs.append(log_copy)
        return all_logs
    
    def print_node_summary(self, tree_index: Optional[int] = None):
        """
        ノードログの要約を出力
        """
        if tree_index is not None:
            if tree_index >= len(self.trees_):
                print(f"Tree index {tree_index} is out of range.")
                return
            logs = self.trees_[tree_index]['tree'].node_logs
            print(f"\n=== Tree {tree_index} Node Summary ===")
        else:
            logs = self.get_node_logs()
            print(f"\n=== All Trees Node Summary ({len(self.trees_)} trees) ===")
        
        if not logs:
            print("No node logs available.")
            return
        
        total_nodes = len(logs)
        leaf_nodes = sum(1 for log in logs if log['is_leaf'])
        print(f"Total nodes: {total_nodes}, Split nodes: {total_nodes - leaf_nodes}, Leaf nodes: {leaf_nodes}")

    def save_logs_to_json(self, file_path: str):
        """
        ノードログをJSONファイルに保存
        """
        all_logs = self.get_node_logs()
        for log in all_logs:
            log['timestamp'] = datetime.now().isoformat()
        with open(file_path, 'w') as json_file:
            json.dump(all_logs, json_file, ensure_ascii=False, indent=4)
        print(f"Node logs saved to {file_path}")

    def _validate_input(self, X: np.ndarray, y_multi: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        入力データの検証と前処理
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if y_multi is not None:
            if y_multi.ndim == 1:
                y_multi = y_multi.reshape(-1, 1)
            if X.shape[0] != y_multi.shape[0]:
                raise ValueError("X and y_multi have different numbers of samples")
            data_n_tasks = y_multi.shape[1]
            # ordinal戦略用: 2列でもOK
            if self.weighting_strategy == 'ordinal':
                if data_n_tasks != 2:
                    raise ValueError("ordinal戦略は2列(click, cv)データのみ対応です")
                self.n_tasks = 2
            elif data_n_tasks == 2 and self.n_tasks == 3:
                y_multi = add_cvr_labels(y_multi)
                self.n_tasks = y_multi.shape[1]
            else:
                self.n_tasks = y_multi.shape[1]
        return X, y_multi
