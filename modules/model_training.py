"""
Model Training Module for Predictive Modeling Dashboard.
Trains multiple supervised models, computes performance metrics, tracks progress,
identifies best performing algorithm, and handles Joblib persistence.
"""

import time
import os
import sys

# Silence Loky physical cores check on Windows
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from modules.model_evaluation import evaluate_classification, evaluate_regression


def train_single_model(
    model_name: str,
    model_info: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    problem_type: str,
    target_classes: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Train an individual model, record latency, predict on test data, and compute metrics.
    """
    model = model_info['model']

    # Record training time
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    # Predictions
    t1 = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - t1

    # Probability estimation for classification
    y_prob = None
    if problem_type == 'Classification':
        if hasattr(model, 'predict_proba'):
            try:
                y_prob = model.predict_proba(X_test)
            except Exception:
                y_prob = None
        elif hasattr(model, 'decision_function'):
            try:
                df_scores = model.decision_function(X_test)
                # Sigmoid normalization for binary
                if df_scores.ndim == 1:
                    probs_pos = 1 / (1 + np.exp(-df_scores))
                    y_prob = np.vstack([1 - probs_pos, probs_pos]).T
            except Exception:
                y_prob = None

        metrics = evaluate_classification(y_test, y_pred, y_prob, target_classes)
    else:
        metrics = evaluate_regression(y_test, y_pred)

    # Feature importances / coefficients if supported
    feature_importance = None
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # For multi-class logistic regression, average across classes
        coef = model.coef_
        if coef.ndim > 1:
            feature_importance = np.mean(np.abs(coef), axis=0)
        else:
            feature_importance = coef

    return {
        'model_name': model_name,
        'model': model,
        'description': model_info.get('description', ''),
        'type': model_info.get('type', ''),
        'train_time_sec': round(train_time, 4),
        'pred_time_sec': round(pred_time, 4),
        'metrics': metrics,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'feature_importance': feature_importance
    }


def train_multiple_models(
    models_dict: Dict[str, Dict[str, Any]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    problem_type: str,
    target_classes: Optional[List[str]] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None
) -> Dict[str, Any]:
    """
    Train a collection of candidate models with live progress updates.
    """
    results = {}
    total = len(models_dict)

    for idx, (name, info) in enumerate(models_dict.items()):
        if progress_callback:
            progress_callback(int((idx / total) * 100), f"Training {name} ({idx+1}/{total})...")

        try:
            res = train_single_model(
                model_name=name,
                model_info=info,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                problem_type=problem_type,
                target_classes=target_classes
            )
            results[name] = res
        except Exception as e:
            results[name] = {
                'model_name': name,
                'error': str(e),
                'metrics': {}
            }

    if progress_callback:
        progress_callback(100, "All models successfully trained!")

    return results


def identify_best_model(results: Dict[str, Any], problem_type: str) -> Optional[str]:
    """
    Determine the top-performing model:
    - Classification: highest F1 Score (tiebreaker: Accuracy)
    - Regression: highest R2 Score (tiebreaker: lowest RMSE)
    """
    valid_results = {k: v for k, v in results.items() if 'metrics' in v and len(v['metrics']) > 0 and 'error' not in v}
    if not valid_results:
        return None

    if problem_type == 'Classification':
        def sort_key(name):
            m = valid_results[name]['metrics']
            f1 = m.get('F1 Score', 0.0)
            acc = m.get('Accuracy', 0.0)
            return (f1, acc)
        best_name = max(valid_results.keys(), key=sort_key)
    else:
        def sort_key(name):
            m = valid_results[name]['metrics']
            r2 = m.get('R2 Score', -999.0)
            rmse = m.get('RMSE', 999999.0)
            return (r2, -rmse)
        best_name = max(valid_results.keys(), key=sort_key)

    return best_name


def save_trained_pipeline(
    filepath: str,
    model: Any,
    model_name: str,
    preprocessor: Any,
    target_encoder: Any,
    feature_cols: List[str],
    numerical_cols: List[str],
    categorical_cols: List[str],
    target_col: str,
    problem_type: str,
    metrics: Dict[str, Any],
    feature_names_out: List[str],
    target_classes: Optional[List[str]] = None
) -> str:
    """
    Bundle trained model, preprocessing pipeline, and full metadata into a single Joblib artifact.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    bundle = {
        'model': model,
        'model_name': model_name,
        'preprocessor': preprocessor,
        'target_encoder': target_encoder,
        'feature_cols': feature_cols,
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols,
        'target_col': target_col,
        'problem_type': problem_type,
        'metrics': metrics,
        'feature_names_out': feature_names_out,
        'target_classes': target_classes,
        'saved_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    joblib.dump(bundle, filepath)
    return filepath


def load_trained_pipeline(filepath: str) -> Dict[str, Any]:
    """
    Load bundled model and pipeline from Joblib file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Saved model file not found at: {filepath}")
    return joblib.load(filepath)
