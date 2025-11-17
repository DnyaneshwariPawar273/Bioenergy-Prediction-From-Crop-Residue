from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from default_values import crop_defaults

NUMERIC_COLS = [
    'Yield_t_ha',
    'Residue_to_Product_Ratio',
    'Residue_Quantity_t',
    'Moisture_%',
    'Calorific_Value_MJ_kg',
]
CATEGORICAL_COLS = ['Crop_Type']
TARGET_COL = 'Predicted_Energy_MJ'


class CropDefaultsImputer(BaseEstimator, TransformerMixin):
    """
    Fill missing RPR, Moisture, CV based on crop-specific defaults.
    Leaves values as-is if present. Unknown crops are left untouched.
    """
    def __init__(self, defaults: Dict[str, Dict[str, float]]):
        self.defaults = defaults

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # Normalize crop names for safety (strip spaces)
        if 'Crop_Type' in X.columns:
            X['Crop_Type'] = X['Crop_Type'].astype(str).str.strip()
        for idx, row in X.iterrows():
            crop = row.get('Crop_Type')
            if crop in self.defaults:
                d = self.defaults[crop]
                if pd.isna(row.get('Residue_to_Product_Ratio')):
                    X.at[idx, 'Residue_to_Product_Ratio'] = d['RPR']
                if pd.isna(row.get('Moisture_%')):
                    X.at[idx, 'Moisture_%'] = d['Moisture']
                if pd.isna(row.get('Calorific_Value_MJ_kg')):
                    X.at[idx, 'Calorific_Value_MJ_kg'] = d['CV']
        return X


class ResidueCalculator(BaseEstimator, TransformerMixin):
    """
    Ensure Residue_Quantity_t exists: if missing/NaN, compute as Yield_t_ha * RPR.
    """
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        need = X['Residue_Quantity_t'].isna() if 'Residue_Quantity_t' in X.columns else pd.Series(True, index=X.index)
        # Ensure columns exist
        if 'Residue_Quantity_t' not in X.columns:
            X['Residue_Quantity_t'] = np.nan
        mask = need & X['Yield_t_ha'].notna() & X['Residue_to_Product_Ratio'].notna()
        X.loc[mask, 'Residue_Quantity_t'] = X.loc[mask, 'Yield_t_ha'] * X.loc[mask, 'Residue_to_Product_Ratio']
        return X


def build_preprocessor() -> Pipeline:
    # Pipeline that applies custom imputations and calculations, then encodes/scales.
    pre_steps = [
        ('defaults', CropDefaultsImputer(crop_defaults)),
        ('residue', ResidueCalculator()),
    ]

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])

    column_transformer = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_COLS),
            ('cat', categorical_transformer, CATEGORICAL_COLS),
        ],
        remainder='drop',
    )

    preprocessor = Pipeline(steps=pre_steps + [('columns', column_transformer)])
    return preprocessor


def train_test_split_processed(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Pipeline]:
    from sklearn.model_selection import train_test_split

    features = CATEGORICAL_COLS + NUMERIC_COLS
    df = df.copy()
    # Keep only needed columns + target if present
    cols = [c for c in features if c in df.columns]
    missing_needed = set(features) - set(cols)
    for c in missing_needed:
        # create columns missing so that custom transformers can compute/fill them
        df[c] = np.nan
    cols = features

    X = df[cols]
    y = df[TARGET_COL].values if TARGET_COL in df.columns else None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train, y_train)
    X_test_proc = preprocessor.transform(X_test)

    return X_train_proc, X_test_proc, y_train, y_test, preprocessor


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def describe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return df[numeric_cols].describe().T
