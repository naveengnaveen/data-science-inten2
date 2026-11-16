"""
Data Loader Module for Predictive Modeling Machine Learning Dashboard.
Handles file ingestion (CSV, Excel), dataset inspection, summary statistics,
and automated problem type detection (Classification vs Regression).
"""

import io
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional, List


def load_dataset(file_or_path: Any) -> pd.DataFrame:
    """
    Load dataset from a file path, file-like object, or Streamlit UploadedFile.
    Supports CSV and Excel (.xlsx, .xls) files.
    """
    if isinstance(file_or_path, str):
        if file_or_path.endswith('.csv'):
            return pd.read_csv(file_or_path)
        elif file_or_path.endswith(('.xlsx', '.xls')):
            return pd.read_excel(file_or_path)
        else:
            # Fallback attempt
            try:
                return pd.read_csv(file_or_path)
            except Exception:
                return pd.read_excel(file_or_path)
    else:
        # File-like object (e.g. Streamlit UploadedFile)
        filename = getattr(file_or_path, 'name', '').lower()
        if filename.endswith(('.xlsx', '.xls')):
            return pd.read_excel(file_or_path)
        else:
            return pd.read_csv(file_or_path)


def get_dataset_overview(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate comprehensive dataset overview statistics:
    rows, columns, memory usage, duplicate count, missing values,
    and column-wise data types.
    """
    total_cells = df.shape[0] * df.shape[1]
    total_missing = int(df.isnull().sum().sum())
    missing_percentage = (total_missing / total_cells * 100) if total_cells > 0 else 0.0
    duplicate_count = int(df.duplicated().sum())

    # Column details table
    col_info = []
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        null_pct = (null_count / len(df) * 100) if len(df) > 0 else 0.0
        n_unique = int(df[col].nunique(dropna=True))
        dtype = str(df[col].dtype)
        col_info.append({
            'Column': col,
            'Data Type': dtype,
            'Missing Values': null_count,
            'Missing (%)': round(null_pct, 2),
            'Unique Values': n_unique
        })

    col_df = pd.DataFrame(col_info)

    # Memory usage in MB
    mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

    return {
        'num_rows': df.shape[0],
        'num_columns': df.shape[1],
        'total_missing': total_missing,
        'missing_percentage': round(missing_percentage, 2),
        'duplicate_count': duplicate_count,
        'memory_mb': round(mem_mb, 2),
        'columns_summary': col_df,
        'numerical_columns': list(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(df.select_dtypes(exclude=[np.number]).columns)
    }


def detect_id_columns(df: pd.DataFrame) -> List[str]:
    """
    Detect likely identifier columns (e.g. ID, CustomerID, index-like)
    that have very high cardinality and are not useful as prediction features.
    """
    id_cols = []
    n_rows = len(df)
    if n_rows == 0:
        return id_cols

    for col in df.columns:
        col_lower = col.lower()
        # Heuristic 1: column name contains id, uuid, key, code
        is_id_name = any(kw in col_lower for kw in ['id', 'uuid', 'guid', 'cust_id', 'customerid'])
        # Heuristic 2: 100% unique values
        is_unique = (df[col].nunique() == n_rows)
        # Heuristic 3: monotonically increasing integer
        is_seq = False
        if pd.api.types.is_integer_dtype(df[col]):
            diffs = df[col].dropna().diff().dropna()
            if len(diffs) > 10 and (diffs == 1).all():
                is_seq = True

        if (is_id_name and is_unique) or is_seq:
            id_cols.append(col)

    return id_cols


def detect_problem_type(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    """
    Intelligently infer whether predicting target_col is a Classification
    or Regression problem, with reasoning, class distribution, or value ranges.
    """
    if target_col not in df.columns:
        return {
            'problem_type': 'Classification',
            'reason': 'Target column not found.',
            'num_classes': 0,
            'classes': []
        }

    target_series = df[target_col].dropna()
    num_unique = target_series.nunique()
    total_samples = len(target_series)

    # Case 1: Non-numeric dtype (object, category, string, bool) -> definitely classification
    if not pd.api.types.is_numeric_dtype(target_series):
        classes = sorted(list(target_series.astype(str).unique()))
        return {
            'problem_type': 'Classification',
            'reason': f"Target column '{target_col}' has non-numeric type ({target_series.dtype}) with {num_unique} unique categories.",
            'num_classes': num_unique,
            'classes': classes[:20],
            'is_binary': num_unique == 2
        }

    # Case 2: Numeric dtype but only 2 unique values (e.g. 0 and 1) -> Binary classification
    if num_unique == 2:
        classes = sorted(list(target_series.unique()))
        return {
            'problem_type': 'Classification',
            'reason': f"Target column '{target_col}' is binary numeric with exactly 2 distinct values ({classes}).",
            'num_classes': 2,
            'classes': classes,
            'is_binary': True
        }

    # Case 3: Numeric dtype with very few unique values (< 10) and low ratio to sample size
    if pd.api.types.is_integer_dtype(target_series) and num_unique <= 10:
        classes = sorted(list(target_series.unique()))
        return {
            'problem_type': 'Classification',
            'reason': f"Target column '{target_col}' has integer values with only {num_unique} discrete categories ({classes}).",
            'num_classes': num_unique,
            'classes': classes,
            'is_binary': False
        }

    # Case 4: Floating point or continuous integers -> Regression
    return {
        'problem_type': 'Regression',
        'reason': f"Target column '{target_col}' is continuous numeric with {num_unique} unique values spanning from {target_series.min()} to {target_series.max()}.",
        'num_classes': None,
        'classes': [],
        'min_val': float(target_series.min()),
        'max_val': float(target_series.max()),
        'mean_val': float(target_series.mean()),
        'std_val': float(target_series.std())
    }
