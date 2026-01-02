"""
ESMM (Entire Space Multi-Task Model) Implementation
===================================================

ESMMモデルの実装。論文「Entire Space Multi-Task Model: An Effective Approach for 
Estimating Post-Click Conversion Rate」に基づく。

Key Features:
- 共有Embedding層を持つマルチタスク学習
- CTRとCTCVRの同時学習によるCVR推定
- サンプルセレクションバイアスとデータスパース性の問題を解決
- 全インプレッションデータでの学習

Architecture:
- Shared Embedding Layer
- CTR Network (MLP)
- CVR Network (MLP) 
- pCTCVR = pCTR × pCVR

Loss Function:
L = L_ctr + L_ctcvr
where L_ctr = BCE(y_click, pCTR)
      L_ctcvr = BCE(y_click_and_conversion, pCTCVR)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import Dict, List, Tuple, Union, Optional, Any
import warnings
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.preprocessing import LabelEncoder
import time
from .config_loader import load_model_config


class ESMM(BaseEstimator, ClassifierMixin):
    """
    ESMM (Entire Space Multi-Task Model) implementation
    
    ESMMは、CTRとCTCVRを同時に学習することで、CVRを正確に推定するモデル。
    共有Embedding層とマルチタスク学習により、サンプルセレクションバイアスと
    データスパース性の問題を解決する。
    
    Parameters:
    -----------
    embedding_dim : int, default=64
        Embedding層の次元数
    mlp_dims : list, default=[360, 200, 80]
        MLP各層の次元数（論文準拠）
    activation : str, default='relu'
        中間層の活性化関数
    dropout_rate : float, default=0.2
        Dropoutの確率
    learning_rate : float, default=0.001
        学習率
    batch_size : int, default=256
        バッチサイズ
    epochs : int, default=100
        エポック数
    early_stopping_patience : int, default=10
        Early stoppingの忍耐パラメータ
    validation_split : float, default=0.2
        検証データの割合
    loss_weights : dict, default=None
        損失関数の重み {'ctr': 1.0, 'ctcvr': 1.0}
    random_state : int, default=42
        ランダムシード
    verbose : int, default=1
        学習時の出力レベル
    """
    
    def __init__(
        self,
        embedding_dim: Optional[int] = None,
        mlp_dims: Optional[List[int]] = None,
        activation: Optional[str] = None,
        dropout_rate: Optional[float] = None,
        learning_rate: Optional[float] = None,
        batch_size: Optional[int] = None,
        epochs: Optional[int] = None,
        early_stopping_patience: Optional[int] = None,
        validation_split: Optional[float] = None,
        loss_weights: Optional[Dict[str, float]] = None,
        random_state: Optional[int] = None,
        verbose: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
        config_mode: Optional[str] = None,
        config_dir: Optional[str] = None,
        use_config: bool = True,
    ):
        defaults = {
            'embedding_dim': 64,
            'mlp_dims': [360, 200, 80],
            'activation': 'relu',
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'batch_size': 256,
            'epochs': 100,
            'early_stopping_patience': 10,
            'validation_split': 0.2,
            'loss_weights': {'ctr': 1.0, 'ctcvr': 1.0},
            'random_state': 42,
            'verbose': 1,
        }

        cfg = load_model_config('esmm', mode=config_mode, config_dir=config_dir) if use_config else {}
        if config:
            cfg.update(config)

        resolved = {**defaults, **cfg}
        explicit_overrides = {
            'embedding_dim': embedding_dim,
            'mlp_dims': mlp_dims,
            'activation': activation,
            'dropout_rate': dropout_rate,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'epochs': epochs,
            'early_stopping_patience': early_stopping_patience,
            'validation_split': validation_split,
            'loss_weights': loss_weights,
            'random_state': random_state,
            'verbose': verbose,
        }
        for key, value in explicit_overrides.items():
            if value is not None:
                resolved[key] = value

        self.embedding_dim = resolved['embedding_dim']
        self.mlp_dims = resolved['mlp_dims']
        self.activation = resolved['activation']
        self.dropout_rate = resolved['dropout_rate']
        self.learning_rate = resolved['learning_rate']
        self.batch_size = resolved['batch_size']
        self.epochs = resolved['epochs']
        self.early_stopping_patience = resolved['early_stopping_patience']
        self.validation_split = resolved['validation_split']
        self.loss_weights = resolved['loss_weights'] or {'ctr': 1.0, 'ctcvr': 1.0}
        self.random_state = resolved['random_state']
        self.verbose = resolved['verbose']
        
        # 学習後に設定される属性
        self.model_ = None
        self.cvr_model_ = None  # CVR推論専用モデル
        self.feature_encoders_ = {}
        self.n_features_ = None
        self.categorical_features_ = None
        self.feature_vocab_sizes_ = {}
        self.history_ = None
        
        # ランダムシード設定
        tf.random.set_seed(self.random_state)
        np.random.seed(self.random_state)
    
    def _prepare_categorical_features(self, X: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        カテゴリカル特徴量の前処理
        
        Parameters:
        -----------
        X : array-like, shape=(n_samples, n_features)
            入力特徴量
            
        Returns:
        --------
        X_encoded : np.ndarray
            エンコード済み特徴量
        feature_info : dict
            特徴量情報
        """
        n_samples, n_features = X.shape
        X_encoded = np.zeros_like(X, dtype=np.int32)
        feature_info = {}
        
        for i in range(n_features):
            feature_name = f'feature_{i}'
            
            if feature_name not in self.feature_encoders_:
                # 新しい特徴量の場合、LabelEncoderを作成
                encoder = LabelEncoder()
                # 未知のカテゴリを0にマッピングするため、0を追加
                unique_values = np.unique(X[:, i])
                encoder.fit(np.concatenate([[0], unique_values]))
                self.feature_encoders_[feature_name] = encoder
                vocab_size = len(encoder.classes_)
            else:
                # 既存の特徴量の場合
                encoder = self.feature_encoders_[feature_name]
                vocab_size = len(encoder.classes_)
            
            # エンコード（未知のカテゴリは0にマッピング）
            try:
                X_encoded[:, i] = encoder.transform(X[:, i])
            except ValueError:
                # 未知のカテゴリがある場合の処理
                encoded_values = []
                for val in X[:, i]:
                    if val in encoder.classes_:
                        encoded_values.append(encoder.transform([val])[0])
                    else:
                        encoded_values.append(0)  # 未知のカテゴリは0
                X_encoded[:, i] = encoded_values
            
            feature_info[feature_name] = {
                'vocab_size': vocab_size,
                'feature_idx': i
            }
        
        return X_encoded, feature_info
    
    def _build_model(self, feature_info: Dict) -> Tuple[Model, Model]:
        """
        ESMMモデルの構築
        
        Parameters:
        -----------
        feature_info : dict
            特徴量情報
            
        Returns:
        --------
        model : tf.keras.Model
            学習用ESMMモデル
        cvr_model : tf.keras.Model
            CVR推論専用モデル
        """
        # 入力層の定義
        feature_inputs = []
        embedding_layers = []
        
        for feature_name, info in feature_info.items():
            vocab_size = info['vocab_size']
            
            # 各特徴量の入力
            feature_input = layers.Input(shape=(1,), name=f'input_{feature_name}')
            feature_inputs.append(feature_input)
            
            # Embedding層（共有）
            embedding = layers.Embedding(
                input_dim=vocab_size,
                output_dim=self.embedding_dim,
                name=f'embedding_{feature_name}'
            )(feature_input)
            # (batch_size, 1, embedding_dim) -> (batch_size, embedding_dim)
            embedding = layers.Flatten()(embedding)
            embedding_layers.append(embedding)
        
        # 共有Embedding層の結合
        if len(embedding_layers) > 1:
            shared_embedding = layers.Concatenate(name='shared_embedding')(embedding_layers)
        else:
            shared_embedding = embedding_layers[0]
        
        # CTR Network (MLP Tower)
        ctr_hidden = shared_embedding
        for i, dim in enumerate(self.mlp_dims):
            ctr_hidden = layers.Dense(
                dim, 
                activation=self.activation,
                name=f'ctr_dense_{i}'
            )(ctr_hidden)
            ctr_hidden = layers.Dropout(self.dropout_rate)(ctr_hidden)
        
        # pCTR出力 - 低いCTRを予測するよう初期化を調整
        pCTR = layers.Dense(
            1, 
            activation='sigmoid', 
            name='pCTR',
            bias_initializer=tf.constant_initializer(-3.0),  # 低いCTR用
            kernel_initializer='glorot_uniform'
        )(ctr_hidden)
        
        # CVR Network (MLP Tower)
        cvr_hidden = shared_embedding
        for i, dim in enumerate(self.mlp_dims):
            cvr_hidden = layers.Dense(
                dim,
                activation=self.activation,
                name=f'cvr_dense_{i}'
            )(cvr_hidden)
            cvr_hidden = layers.Dropout(self.dropout_rate)(cvr_hidden)
        
        # pCVR出力 - CVRも低めに初期化
        pCVR = layers.Dense(
            1, 
            activation='sigmoid', 
            name='pCVR',
            bias_initializer=tf.constant_initializer(-1.0),  # CVR用
            kernel_initializer='glorot_uniform'
        )(cvr_hidden)
        
        # pCTCVR = pCTR × pCVR
        pCTCVR = layers.Multiply(name='pCTCVR')([pCTR, pCVR])
        
        # 学習用モデル（CTRとCTCVRを出力）
        model = Model(
            inputs=feature_inputs,
            outputs={'pCTR': pCTR, 'pCTCVR': pCTCVR},
            name='ESMM'
        )
        
        # CVR推論専用モデル（pCVRのみを出力）
        cvr_model = Model(
            inputs=feature_inputs,
            outputs={'pCVR': pCVR},
            name='ESMM_CVR'
        )
        
        return model, cvr_model
    
    def _prepare_labels(self, y_multi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        ラベルの準備
        
        Parameters:
        -----------
        y_multi : array-like, shape=(n_samples, 2)
            [Click, Conversion] のラベル
            
        Returns:
        --------
        y_ctr : np.ndarray
            CTRラベル（クリック有無）
        y_ctcvr : np.ndarray
            CTCVRラベル（クリックかつコンバージョン有無）
        """
        y_click = y_multi[:, 0]  # Click
        y_conversion = y_multi[:, 1]  # Conversion
        
        y_ctr = y_click.astype(np.float32)
        y_ctcvr = (y_click * y_conversion).astype(np.float32)  # Click AND Conversion
        
        return y_ctr, y_ctcvr
    
    def _prepare_input_data(self, X_encoded: np.ndarray) -> Dict[str, np.ndarray]:
        """
        モデル入力用のデータ準備（辞書形式）
        
        Parameters:
        -----------
        X_encoded : np.ndarray
            エンコード済み特徴量
            
        Returns:
        --------
        input_data : dict
            各特徴量ごとの入力データ辞書
        """
        n_samples, n_features = X_encoded.shape
        input_data = {}
        
        # feature_info_から特徴量名を取得
        if hasattr(self, 'feature_info_'):
            feature_names = list(self.feature_info_.keys())
        else:
            feature_names = [f'feature_{i}' for i in range(n_features)]
        
        for i, feature_name in enumerate(feature_names):
            if i < n_features:
                # (n_samples,) -> (n_samples, 1)
                feature_data = X_encoded[:, i].reshape(-1, 1)
                input_data[f'input_{feature_name}'] = feature_data
        
        return input_data
    
    def fit(self, X: np.ndarray, y_multi: np.ndarray, **fit_params) -> 'ESMM':
        """
        ESMMモデルの学習
        
        Parameters:
        -----------
        X : array-like, shape=(n_samples, n_features)
            入力特徴量
        y_multi : array-like, shape=(n_samples, 2)
            [Click, Conversion] のラベル
        **fit_params : dict
            追加の学習パラメータ
            
        Returns:
        --------
        self : ESMM
            学習済みモデル
        """
        # 入力データの検証
        X, y_multi = check_X_y(X, y_multi, multi_output=True)
        
        if y_multi.shape[1] != 2:
            raise ValueError("y_multi must have shape (n_samples, 2) for [Click, Conversion]")
        
        self.n_features_ = X.shape[1]
        
        if self.verbose > 0:
            print("ESMM Model Training Started...")
            print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
        
        # カテゴリカル特徴量の前処理
        X_encoded, feature_info = self._prepare_categorical_features(X)
        self.feature_vocab_sizes_ = {name: info['vocab_size'] 
                                   for name, info in feature_info.items()}
        self.feature_info_ = feature_info  # 特徴量情報を保存
        
        # ラベルの準備
        y_ctr, y_ctcvr = self._prepare_labels(y_multi)
        
        # モデル構築
        self.model_, self.cvr_model_ = self._build_model(feature_info)
        
        # コンパイル
        self.model_.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss={
                'pCTR': 'binary_crossentropy',
                'pCTCVR': 'binary_crossentropy'
            },
            loss_weights=self.loss_weights,
            metrics={
                'pCTR': ['accuracy', 'AUC'],
                'pCTCVR': ['accuracy', 'AUC']
            }
        )
        
        if self.verbose > 0:
            print("\nModel Architecture:")
            self.model_.summary()
        
        # 入力データの準備
        input_data = self._prepare_input_data(X_encoded)
        target_data = {'pCTR': y_ctr, 'pCTCVR': y_ctcvr}
        
        # コールバック設定
        callbacks = []
        if self.early_stopping_patience > 0:
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.early_stopping_patience,
                restore_best_weights=True,
                verbose=self.verbose
            )
            callbacks.append(early_stopping)
        
        # 学習実行
        start_time = time.time()
        
        self.history_ = self.model_.fit(
            input_data,
            target_data,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=self.validation_split,
            callbacks=callbacks,
            verbose=self.verbose,
            **fit_params
        )
        
        training_time = time.time() - start_time
        
        if self.verbose > 0:
            print(f"\nTraining completed in {training_time:.2f} seconds")
            
            # 最終損失の表示
            final_loss = self.history_.history['loss'][-1]
            final_val_loss = self.history_.history['val_loss'][-1] if 'val_loss' in self.history_.history else None
            print(f"Final training loss: {final_loss:.4f}")
            if final_val_loss:
                print(f"Final validation loss: {final_val_loss:.4f}")
        
        return self
    
    def predict_cvr(self, X: np.ndarray) -> np.ndarray:
        """
        CVR予測（pCVRを直接取得）
        
        Parameters:
        -----------
        X : array-like, shape=(n_samples, n_features)
            入力特徴量
            
        Returns:
        --------
        pCVR : np.ndarray, shape=(n_samples,)
            CVR予測値
        """
        if self.cvr_model_ is None:
            raise ValueError("Model must be fitted before prediction")
        
        # 入力データの検証
        X = check_array(X)
        
        # カテゴリカル特徴量のエンコード
        X_encoded, _ = self._prepare_categorical_features(X)
        
        # 入力データの準備
        input_data = self._prepare_input_data(X_encoded)
        
        # CVR予測実行
        predictions = self.cvr_model_.predict(input_data, batch_size=self.batch_size, verbose=0)
        
        return predictions['pCVR'].flatten()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        3タスク確率予測（CTR, CTCVR, CVR）
        MTGBDTとの互換性のために、[pCTR, pCTCVR, pCVR] の形式で返す。
        
        Parameters:
        -----------
        X : array-like, shape=(n_samples, n_features)
            入力特徴量
            
        Returns:
        --------
        predictions : np.ndarray, shape=(n_samples, 3)
            [pCTR, pCTCVR, pCVR] の確率予測値
        """
        if self.model_ is None or self.cvr_model_ is None:
            raise ValueError("Model must be fitted before prediction")
        
        # 入力データの検証
        X = check_array(X)
        
        # カテゴリカル特徴量のエンコード
        X_encoded, _ = self._prepare_categorical_features(X)
        
        # 入力データの準備
        input_data = self._prepare_input_data(X_encoded)
        
        # メインモデルから pCTR, pCTCVR を取得
        main_predictions = self.model_.predict(input_data, batch_size=self.batch_size, verbose=0)
        pCTR = main_predictions['pCTR'].flatten()
        pCTCVR = main_predictions['pCTCVR'].flatten()
        
        # CVRは計算で求める: pCVR = pCTCVR / pCTR （ゼロ除算対策付き）
        pCVR = np.where(pCTR > 1e-8, pCTCVR / pCTR, 0.0)
        # pCVRを0-1範囲にクリップ
        pCVR = np.clip(pCVR, 0.0, 1.0)
        
        # [pCTR, pCTCVR, pCVR] の形式で返却（3タスク対応）
        return np.column_stack([pCTR, pCTCVR, pCVR])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """互換性のため predict は predict_proba の結果を返す。"""
        return self.predict_proba(X)
    
    def get_training_history(self) -> Optional[Dict]:
        """
        学習履歴の取得
        
        Returns:
        --------
        history : dict or None
            学習履歴
        """
        if self.history_ is None:
            return None
        
        return self.history_.history
    
    def save_model(self, filepath: str) -> None:
        """
        モデルの保存
        
        Parameters:
        -----------
        filepath : str
            保存先ファイルパス
        """
        if self.model_ is None:
            raise ValueError("Model must be fitted before saving")
        
        # メインモデルの保存
        self.model_.save(f"{filepath}_main.h5")
        
        # CVRモデルの保存
        self.cvr_model_.save(f"{filepath}_cvr.h5")
        
        # エンコーダーの保存（簡易的にnpzで保存）
        encoder_data = {}
        for name, encoder in self.feature_encoders_.items():
            encoder_data[f"{name}_classes"] = encoder.classes_
        
        np.savez(f"{filepath}_encoders.npz", **encoder_data)
        
        if self.verbose > 0:
            print(f"Model saved to {filepath}_main.h5, {filepath}_cvr.h5, {filepath}_encoders.npz")
    
    def load_model(self, filepath: str) -> 'ESMM':
        """
        モデルの読み込み
        
        Parameters:
        -----------
        filepath : str
            モデルファイルパス
            
        Returns:
        --------
        self : ESMM
            読み込み済みモデル
        """
        # メインモデルの読み込み
        self.model_ = keras.models.load_model(f"{filepath}_main.h5")
        
        # CVRモデルの読み込み
        self.cvr_model_ = keras.models.load_model(f"{filepath}_cvr.h5")
        
        # エンコーダーの読み込み
        encoder_data = np.load(f"{filepath}_encoders.npz")
        self.feature_encoders_ = {}
        
        feature_names = set([key.replace('_classes', '') for key in encoder_data.keys()])
        for name in feature_names:
            encoder = LabelEncoder()
            encoder.classes_ = encoder_data[f"{name}_classes"]
            self.feature_encoders_[name] = encoder
        
        if self.verbose > 0:
            print(f"Model loaded from {filepath}")
        
        return self


# gbdt.pyとの互換性のためのエイリアス
class ESMMModel(ESMM):
    """gbdt.pyとの命名互換性のためのエイリアス"""
    pass
