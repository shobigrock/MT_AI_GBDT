"""
MT-GBM基底クラスモジュール

このモジュールは、マルチタスク勾配ブースティングマシン（MT-GBM）の
抽象基底クラスを提供します。すべてのMT-GBM実装はこのクラスを継承します。
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any

from .utils import TASK_ORDER, TASK_INDEX


class MTGBMBase(ABC):
    """
    マルチタスク勾配ブースティングマシン（MT-GBM）の抽象基底クラス
    
    このクラスは、すべてのMT-GBM実装に共通するインターフェースを定義します。
    各実装（スクラッチGBDT、XGBoost、LightGBM）はこのクラスを継承し、
    抽象メソッドを実装する必要があります。
    
    Attributes:
    -----------
    n_estimators : int
        ブースティング反復回数（木の数）
    learning_rate : float
        学習率（各木の寄与度）
    max_depth : int
        各木の最大深さ
    n_tasks : int
        タスク数
    models : list
        学習済みモデル（木）のリスト
    """
    
    def __init__(self, 
                 n_estimators: int = 100, 
                 learning_rate: float = 0.1, 
                 max_depth: int = 3,
                 random_state: Optional[int] = None,
                 **kwargs):
        """
        初期化メソッド
        
        Parameters:
        -----------
        n_estimators : int, default=100
            ブースティング反復回数（木の数）
        learning_rate : float, default=0.1
            学習率（各木の寄与度）
        max_depth : int, default=3
            各木の最大深さ
        random_state : int, optional
            乱数シード
        **kwargs : dict
            追加のパラメータ
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.n_tasks = None
        self.models = []
        self.task_order = TASK_ORDER
        
        # 追加のパラメータを設定
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @abstractmethod
    def fit(self, X: np.ndarray, y_multi: np.ndarray, **kwargs) -> 'MTGBMBase':
        """
        マルチタスクデータでモデルを学習
        
        Parameters:
        -----------
        X : array-like, shape=(n_samples, n_features)
            入力特徴量
        y_multi : array-like, shape=(n_samples, n_tasks)
            複数タスクのターゲット値
        **kwargs : dict
            追加のパラメータ
            
        Returns:
        --------
        self : MTGBMBase
            学習済みモデル
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        学習済みモデルで予測
        
        Parameters:
        -----------
        X : array-like, shape=(n_samples, n_features)
            入力特徴量
            
        Returns:
        --------
        y_pred : array-like, shape=(n_samples, n_tasks)
            各タスクの予測値
        """
        pass

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """分類器用の確率予測。未実装の場合はサブクラスでオーバーライド。"""
        raise NotImplementedError("predict_proba is not implemented for this model.")
    
    def _validate_input(self, X: np.ndarray, y_multi: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        入力データの検証と前処理
        
        Parameters:
        -----------
        X : array-like
            入力特徴量
        y_multi : array-like, optional
            複数タスクのターゲット値
            
        Returns:
        --------
        X : np.ndarray
            検証・変換後の入力特徴量
        y_multi : np.ndarray or None
            検証・変換後のターゲット値
        """
        # Xを2次元配列に変換
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # y_multiが指定されている場合
        if y_multi is not None:
            # y_multiを2次元配列に変換
            if y_multi.ndim == 1:
                y_multi = y_multi.reshape(-1, 1)
            if np.any(~np.isfinite(y_multi)):
                raise ValueError("y_multi contains NaN or infinite values")
            if np.any(~np.isfinite(X)):
                raise ValueError("X contains NaN or infinite values")
                
            # サンプル数の一致を確認
            if X.shape[0] != y_multi.shape[0]:
                raise ValueError(f"X ({X.shape[0]} samples) and y_multi ({y_multi.shape[0]} samples) have different numbers of samples")
            
            # タスク数を設定
            if self.n_tasks is None:
                self.n_tasks = y_multi.shape[1]
            elif self.n_tasks != y_multi.shape[1]:
                raise ValueError(f"y_multi has {y_multi.shape[1]} tasks, but model was initialized with {self.n_tasks} tasks")
                    
        return X, y_multi
    
    def evaluate(self, X: np.ndarray, y_multi: np.ndarray, metrics: List[str] = ['mse']) -> Dict[str, np.ndarray]:
        """
        モデルの評価
        
        Parameters:
        -----------
        X : array-like, shape=(n_samples, n_features)
            入力特徴量
        y_multi : array-like, shape=(n_samples, n_tasks)
            複数タスクの真のターゲット値
        metrics : list of str, default=['mse']
            使用する評価指標のリスト
            
        Returns:
        --------
        results : dict
            各タスクの各評価指標の値
        """
        # 入力検証
        X, y_multi = self._validate_input(X, y_multi)
        
        # 予測
        y_pred = self.predict(X)
        
        # 結果格納用辞書
        results = {}
        
        # 各評価指標を計算
        for metric in metrics:
            if metric.lower() == 'mse':
                # 平均二乗誤差
                task_mse = np.mean((y_multi - y_pred) ** 2, axis=0)
                results['mse'] = task_mse
                results['mse_avg'] = np.mean(task_mse)
            
            elif metric.lower() == 'rmse':
                # 平方根平均二乗誤差
                task_rmse = np.sqrt(np.mean((y_multi - y_pred) ** 2, axis=0))
                results['rmse'] = task_rmse
                results['rmse_avg'] = np.mean(task_rmse)
            
            elif metric.lower() == 'mae':
                # 平均絶対誤差
                task_mae = np.mean(np.abs(y_multi - y_pred), axis=0)
                results['mae'] = task_mae
                results['mae_avg'] = np.mean(task_mae)
            
            elif metric.lower() == 'r2':
                # 決定係数
                y_mean = np.mean(y_multi, axis=0)
                ss_tot = np.sum((y_multi - y_mean.reshape(1, -1)) ** 2, axis=0)
                ss_res = np.sum((y_multi - y_pred) ** 2, axis=0)
                task_r2 = 1 - (ss_res / ss_tot)
                results['r2'] = task_r2
                results['r2_avg'] = np.mean(task_r2)
            
            else:
                raise ValueError(f"Unknown metric: {metric}")
        
        return results
    
    def get_params(self) -> Dict[str, Any]:
        """
        モデルパラメータの取得
        
        Returns:
        --------
        params : dict
            モデルパラメータ
        """
        return {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'random_state': self.random_state,
            'n_tasks': self.n_tasks,
            'task_order': self.task_order
        }
    
    def set_params(self, **params) -> 'MTGBMBase':
        """
        モデルパラメータの設定
        
        Parameters:
        -----------
        **params : dict
            設定するパラメータ
            
        Returns:
        --------
        self : MTGBMBase
            パラメータを更新したモデル
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")
        return self
