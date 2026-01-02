import numpy as np
from .config_loader import load_model_config
from .gbdt_proto import MTGBDT
from .utils import add_cvr_labels, TASK_INDEX

class STGBDTBaseline:
    """
    STGBDT（自前実装）を用いた3段階ベースラインモデル
    CTR、CTCVR、CVRを別々に学習する3タスク対応版
    MTGBDTと同一アルゴリズムで実験条件を統一
    """
    
    def __init__(
        self,
        n_estimators=None,  # 元に戻す: 50 → 10
        learning_rate=None,
        max_depth=None,
        min_samples_split=None,
        min_samples_leaf=None,
        random_state=None,
        config=None,
        config_mode=None,
        config_dir=None,
        use_config: bool = True,
    ):
        """Load STGBDT hyperparameters from configs/model/stgbdt.* when available."""
        defaults = {
            "n_estimators": 10,
            "learning_rate": 0.3,
            "max_depth": 2,
            "min_samples_split": 20,
            "min_samples_leaf": 10,
            "random_state": 42,
        }

        cfg = load_model_config("stgbdt", mode=config_mode, config_dir=config_dir) if use_config else {}
        if config:
            cfg.update(config)

        resolved = {**defaults, **cfg}
        overrides = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "random_state": random_state,
        }
        for key, value in overrides.items():
            if value is not None:
                resolved[key] = value

        self.n_estimators = resolved["n_estimators"]
        self.learning_rate = resolved["learning_rate"]
        self.max_depth = resolved["max_depth"]
        self.min_samples_split = resolved["min_samples_split"]
        self.min_samples_leaf = resolved["min_samples_leaf"]
        self.random_state = resolved["random_state"]
        
        # 3つのモデル
        self.ctr_model = None  # 全データでCTR予測
        self.ctcvr_model = None  # 全データでCTCVR予測
        self.conditional_cvr_model = None  # クリックデータで条件付きCVR予測
        
    def fit(self, X, y_multi):
        """
        3段階でモデルを学習
        
        Parameters:
        -----------
        X : array-like, shape=(n_samples, n_features)
            入力特徴量
        y_multi : array-like, shape=(n_samples, 2)
            [Click, Conversion] のラベル（Criteoデータセット形式）
        """
        # [Click, CV] → [CTR, CTCVR, CVR] に変換
        y_3task = add_cvr_labels(y_multi)
        
        y_ctr = y_3task[:, TASK_INDEX["ctr"]]      # CTRタスク
        y_ctcvr = y_3task[:, TASK_INDEX["ctcvr"]]  # CTCVRタスク
        y_cvr = y_3task[:, TASK_INDEX["cvr"]]      # CVRタスク（NaN含む）
        
        # 第1段階: 全データでCTR予測モデル
        self.ctr_model = MTGBDT(
            n_tasks=1,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            loss="logloss",
            random_state=self.random_state
        )
        self.ctr_model.fit(X, y_ctr)
        
        # 第2段階: 全データでCTCVR予測モデル
        self.ctcvr_model = MTGBDT(
            n_tasks=1,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            loss="logloss",
            random_state=self.random_state
        )
        self.ctcvr_model.fit(X, y_ctcvr)
        
        # 第3段階: クリックデータのみで条件付きCVR予測モデル
        click_mask = y_ctr == 1  # Click=1（クリック）のデータのみ
        if np.sum(click_mask) == 0:
            # ダミーモデル（常に0を予測）
            self.conditional_cvr_model = None
        else:
            X_click = X[click_mask]
            # クリックした人のコンバージョン状況 = 条件付きCVR
            y_cvr_click = y_cvr[click_mask]
            
            # NaNを除去（クリックデータではNaNはないはず）
            valid_mask = ~np.isnan(y_cvr_click)
            if np.sum(valid_mask) == 0:
                self.conditional_cvr_model = None
            else:
                X_click_valid = X_click[valid_mask]
                y_cvr_click_valid = y_cvr_click[valid_mask]
                
                if len(np.unique(y_cvr_click_valid)) > 1:  # CVRに変動がある場合のみ学習
                    self.conditional_cvr_model = MTGBDT(
                        n_tasks=1,
                        n_estimators=self.n_estimators,
                        learning_rate=self.learning_rate,
                        max_depth=self.max_depth,
                        min_samples_split=self.min_samples_split,
                        min_samples_leaf=self.min_samples_leaf,
                        loss="logloss",
                        random_state=self.random_state
                    )
                    self.conditional_cvr_model.fit(X_click_valid, y_cvr_click_valid)
                else:
                    self.conditional_cvr_model = None
                    self.cvr_mean = y_cvr_click_valid.mean()
    
    def predict_proba(self, X):
        """3タスク確率予測: [CTR, CTCVR, CVR]"""
        n_samples = X.shape[0]
        
        # CTR予測（確率）
        ctr_proba = self.ctr_model.predict_proba(X)[:, 0]  # シングルタスクなので[:, 0]
        
        # CTCVR予測（確率）
        ctcvr_proba = self.ctcvr_model.predict_proba(X)[:, 0]  # シングルタスクなので[:, 0]
        
        # 条件付きCVR予測
        if self.conditional_cvr_model is not None:
            conditional_cvr_proba = self.conditional_cvr_model.predict_proba(X)[:, 0]  # シングルタスクなので[:, 0]
        else:
            # モデルがない場合は平均値または0を使用
            if hasattr(self, 'cvr_mean'):
                conditional_cvr_proba = np.full(n_samples, self.cvr_mean)
            else:
                conditional_cvr_proba = np.zeros(n_samples)
        
        # [CTR, CTCVR, CVR] の形式で返却（3タスク対応）
        y_pred = np.column_stack([ctr_proba, ctcvr_proba, conditional_cvr_proba])
        
        return y_pred

    def predict(self, X):
        """互換性のため predict は predict_proba を返す"""
        return self.predict_proba(X)