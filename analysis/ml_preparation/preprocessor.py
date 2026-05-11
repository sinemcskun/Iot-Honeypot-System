import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler


class MLPreprocessor:
    SKEW_THRESHOLD = 2.0
    LOG_PREFIX = "log_"

    def __init__(self):
        self.scaler = StandardScaler()
        self._log_transformed = []

    def fit_transform(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        df = df.copy()
        df = self._clean(df, feature_cols)
        df = self._log_transform_skewed(df, feature_cols)
        df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        print(f"[preprocessor] fit_transform: {len(feature_cols)} features scaled.")
        return df

    def transform(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        df = df.copy()
        df = self._clean(df, feature_cols)
        df = self._apply_log_transforms(df)
        df[feature_cols] = self.scaler.transform(df[feature_cols])
        return df

    def save_scaler(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "scaler": self.scaler,
                "log_transformed": self._log_transformed,
            }, f)
        print(f"[preprocessor] Scaler -> {path}")

    @staticmethod
    def load_scaler(path: str) -> "MLPreprocessor":
        with open(path, "rb") as f:
            data = pickle.load(f)
        proc = MLPreprocessor()
        proc.scaler = data["scaler"]
        proc._log_transformed = data["log_transformed"]
        return proc

    @staticmethod
    def _clean(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:   
        for col in feature_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
                df[col] = df[col].replace([np.inf, -np.inf], 0.0)
        return df

    def _log_transform_skewed(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        self._log_transformed = []
        for col in feature_cols:
            if col.startswith(self.LOG_PREFIX):
                continue
            if col in df.columns and df[col].dtype in ["float64", "int64"]:
                skew = df[col].skew()
                if abs(skew) > self.SKEW_THRESHOLD:
                    df[col] = np.log1p(df[col].clip(lower=0))
                    self._log_transformed.append(col)
        if self._log_transformed:
            print(f"[preprocessor] Log-transform: {self._log_transformed}")
        return df

    def _apply_log_transforms(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self._log_transformed:
            if col in df.columns:
                df[col] = np.log1p(df[col].clip(lower=0))
        return df
