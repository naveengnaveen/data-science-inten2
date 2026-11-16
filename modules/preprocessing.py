"""
Data Preprocessing Module for Predictive Modeling Machine Learning Dashboard.
Constructs leak-free Scikit-Learn ColumnTransformers and Pipelines for:
- Missing value imputation (Numerical & Categorical)
- Deduplication
- Categorical encoding (One-Hot Encoding or Ordinal Encoding)
- Numerical feature scaling (StandardScaler, MinMaxScaler, RobustScaler)
- Outlier detection & IQR clipping
- Train-test splitting
- Target encoding for classification
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    OneHotEncoder,
    OrdinalEncoder,
    LabelEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


class IQROutlierClipper(BaseEstimator, TransformerMixin):
    """
    Leak-free custom transformer that learns IQR boundaries on training data
    and clips outliers to [Q1 - factor * IQR, Q3 + factor * IQR].
    """
    def __init__(self, factor: float = 1.5):
        self.factor = factor
        self.lower_bounds_ = None
        self.upper_bounds_ = None

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=float)
        q25 = np.nanpercentile(X_arr, 25, axis=0)
        q75 = np.nanpercentile(X_arr, 75, axis=0)
        iqr = q75 - q25
        self.lower_bounds_ = q25 - (self.factor * iqr)
        self.upper_bounds_ = q75 + (self.factor * iqr)
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=float).copy()
        if self.lower_bounds_ is not None and self.upper_bounds_ is not None:
            # Clip each column
            for col_idx in range(X_arr.shape[1]):
                lb = self.lower_bounds_[col_idx]
                ub = self.upper_bounds_[col_idx]
                if not np.isnan(lb) and not np.isnan(ub) and lb < ub:
                    X_arr[:, col_idx] = np.clip(X_arr[:, col_idx], lb, ub)
        return X_arr


def detect_outliers_iqr(df: pd.DataFrame, num_cols: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Analyze numerical columns for outliers using the IQR technique.
    Returns counts, percentage, and bounding thresholds for each column.
    """
    outlier_info = {}
    for col in num_cols:
        series = df[col].dropna()
        if len(series) < 5:
            continue
        q25 = series.quantile(0.25)
        q75 = series.quantile(0.75)
        iqr = q75 - q25
        lb = q25 - 1.5 * iqr
        ub = q75 + 1.5 * iqr
        outlier_mask = (series < lb) | (series > ub)
        outlier_count = int(outlier_mask.sum())
        outlier_info[col] = {
            'q25': round(float(q25), 3),
            'q75': round(float(q75), 3),
            'iqr': round(float(iqr), 3),
            'lower_bound': round(float(lb), 3),
            'upper_bound': round(float(ub), 3),
            'outlier_count': outlier_count,
            'outlier_pct': round((outlier_count / len(series)) * 100, 2)
        }
    return outlier_info


def build_preprocessor_pipeline(
    numerical_cols: List[str],
    categorical_cols: List[str],
    impute_num_strategy: str = 'median',
    impute_cat_strategy: str = 'most_frequent',
    scaling_method: str = 'StandardScaler',
    encoding_method: str = 'One-Hot Encoding',
    handle_outliers: bool = True
) -> ColumnTransformer:
    """
    Create a robust, leak-free Scikit-Learn ColumnTransformer pipeline.
    """
    transformers = []

    # 1. Numerical Pipeline
    if len(numerical_cols) > 0:
        num_steps = [
            ('imputer', SimpleImputer(strategy=impute_num_strategy))
        ]
        if handle_outliers:
            num_steps.append(('outlier_clipper', IQROutlierClipper(factor=1.5)))

        if scaling_method == 'StandardScaler':
            num_steps.append(('scaler', StandardScaler()))
        elif scaling_method == 'MinMaxScaler':
            num_steps.append(('scaler', MinMaxScaler()))
        elif scaling_method == 'RobustScaler':
            num_steps.append(('scaler', RobustScaler()))
        # 'None' leaves data unscaled

        num_pipeline = Pipeline(num_steps)
        transformers.append(('num', num_pipeline, numerical_cols))

    # 2. Categorical Pipeline
    if len(categorical_cols) > 0:
        cat_steps = [
            ('imputer', SimpleImputer(strategy=impute_cat_strategy, fill_value='Missing'))
        ]
        if encoding_method == 'One-Hot Encoding':
            cat_steps.append(('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)))
        else:
            cat_steps.append(('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)))

        cat_pipeline = Pipeline(cat_steps)
        transformers.append(('cat', cat_pipeline, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    return preprocessor


def extract_feature_names(preprocessor: ColumnTransformer, numerical_cols: List[str], categorical_cols: List[str]) -> List[str]:
    """
    Extract readable feature names after ColumnTransformer transformation.
    """
    feature_names = []
    try:
        # Scikit-learn >= 1.0 has get_feature_names_out
        raw_names = list(preprocessor.get_feature_names_out())
        # Clean prefix like 'num__' and 'cat__'
        cleaned = [name.replace('num__', '').replace('cat__', '') for name in raw_names]
        return cleaned
    except Exception:
        # Fallback manual reconstruction
        for col in numerical_cols:
            feature_names.append(col)
        for col in categorical_cols:
            feature_names.append(f"{col}_encoded")
        return feature_names


def preprocess_and_split(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    problem_type: str,
    test_size: float = 0.20,
    random_state: int = 42,
    remove_duplicates: bool = True,
    impute_num_strategy: str = 'median',
    impute_cat_strategy: str = 'most_frequent',
    scaling_method: str = 'StandardScaler',
    encoding_method: str = 'One-Hot Encoding',
    handle_outliers: bool = True
) -> Dict[str, Any]:
    """
    End-to-end preprocessing pipeline:
    1. Removes duplicates if requested.
    2. Separates X and y.
    3. Handles target encoding if Classification.
    4. Splits into train and test sets BEFORE any fitting to prevent data leakage.
    5. Fits preprocessor ONLY on X_train, then transforms X_train and X_test.
    6. Returns preprocessed arrays, fitted transformers, and preprocessing metadata.
    """
    df_clean = df.copy()

    # Step 1: Remove duplicates
    initial_rows = len(df_clean)
    if remove_duplicates:
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    duplicates_removed = initial_rows - len(df_clean)

    # Step 2: Drop rows where target is NaN
    df_clean = df_clean.dropna(subset=[target_col]).reset_index(drop=True)

    # Separate X and y
    X = df_clean[feature_cols].copy()
    y_raw = df_clean[target_col].copy()

    # Determine numerical and categorical columns from selected features
    num_cols = list(X.select_dtypes(include=[np.number]).columns)
    cat_cols = list(X.select_dtypes(exclude=[np.number]).columns)

    # Outlier statistics on raw data for user reporting
    outlier_stats = detect_outliers_iqr(X, num_cols)

    # Step 3: Target Encoding for Classification
    target_encoder = None
    target_classes = None

    if problem_type == 'Classification':
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y_raw)
        target_classes = list(target_encoder.classes_)
    else:
        y = y_raw.astype(float).values

    # Step 4: Train / Test Split
    stratify = y if (problem_type == 'Classification' and len(np.unique(y)) > 1) else None
    try:
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )
    except Exception:
        # In case stratify fails due to single-member class
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state
        )

    # Step 5: Fit Preprocessor strictly on training data
    preprocessor = build_preprocessor_pipeline(
        numerical_cols=num_cols,
        categorical_cols=cat_cols,
        impute_num_strategy=impute_num_strategy,
        impute_cat_strategy=impute_cat_strategy,
        scaling_method=scaling_method,
        encoding_method=encoding_method,
        handle_outliers=handle_outliers
    )

    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    feature_names_out = extract_feature_names(preprocessor, num_cols, cat_cols)

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_raw': X_train_raw,
        'X_test_raw': X_test_raw,
        'y_train_raw': y_raw.iloc[X_train_raw.index],
        'y_test_raw': y_raw.iloc[X_test_raw.index],
        'preprocessor': preprocessor,
        'target_encoder': target_encoder,
        'target_classes': target_classes,
        'feature_names_out': feature_names_out,
        'numerical_cols': num_cols,
        'categorical_cols': cat_cols,
        'feature_cols': feature_cols,
        'target_col': target_col,
        'problem_type': problem_type,
        'test_size': test_size,
        'duplicates_removed': duplicates_removed,
        'outlier_stats': outlier_stats,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'scaling_method': scaling_method,
        'encoding_method': encoding_method,
        'impute_num_strategy': impute_num_strategy,
        'impute_cat_strategy': impute_cat_strategy,
        'handle_outliers': handle_outliers
    }
