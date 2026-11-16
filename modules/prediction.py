"""
Prediction Module for Predictive Modeling Dashboard.
Supports single-record manual prediction with dynamic form fields & validation,
confidence probability breakdowns, and bulk batch predictions from CSV/Excel.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple


def predict_single_record(
    input_dict: Dict[str, Any],
    model: Any,
    preprocessor: Any,
    target_encoder: Optional[Any],
    problem_type: str,
    feature_cols: List[str]
) -> Dict[str, Any]:
    """
    Run prediction on a single user-input record dictionary.
    Transforms raw input via preprocessor pipeline and invokes trained model.
    """
    # Create single-row DataFrame with exact columns
    input_df = pd.DataFrame([input_dict])[feature_cols]

    # Transform through fitted preprocessor
    X_proc = preprocessor.transform(input_df)

    # Predict
    raw_pred = model.predict(X_proc)[0]

    # Format result
    probabilities = None
    prob_dict = {}

    if problem_type == 'Classification':
        # Decode target label if encoder present
        if target_encoder is not None:
            pred_label = target_encoder.inverse_transform([int(raw_pred)])[0]
            classes = list(target_encoder.classes_)
        else:
            pred_label = str(raw_pred)
            classes = [str(c) for c in getattr(model, 'classes_', [0, 1])]

        # Probabilities if supported
        if hasattr(model, 'predict_proba'):
            try:
                probs = model.predict_proba(X_proc)[0]
                probabilities = probs.tolist()
                for cls, prob in zip(classes, probs):
                    prob_dict[str(cls)] = round(float(prob) * 100, 2)
            except Exception:
                probabilities = None

        return {
            'prediction': pred_label,
            'prediction_raw': raw_pred,
            'probabilities': prob_dict,
            'classes': classes,
            'problem_type': 'Classification'
        }
    else:
        # Regression
        return {
            'prediction': round(float(raw_pred), 4),
            'prediction_formatted': f"{raw_pred:,.2f}",
            'problem_type': 'Regression'
        }


def predict_batch(
    df_batch: pd.DataFrame,
    model: Any,
    preprocessor: Any,
    target_encoder: Optional[Any],
    problem_type: str,
    feature_cols: List[str],
    target_col: str
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Run batch prediction across an entire DataFrame.
    Validates required columns, produces predictions, and appends output columns.
    """
    # Validate missing columns
    missing_cols = [c for c in feature_cols if c not in df_batch.columns]
    if missing_cols:
        return df_batch, f"Uploaded file is missing required feature columns: {', '.join(missing_cols)}"

    # Slice feature columns
    X_input = df_batch[feature_cols].copy()

    # Preprocess
    try:
        X_proc = preprocessor.transform(X_input)
    except Exception as e:
        return df_batch, f"Preprocessing failed during batch transformation: {str(e)}"

    # Predictions
    raw_preds = model.predict(X_proc)
    result_df = df_batch.copy()

    pred_col_name = f"Predicted_{target_col}"

    if problem_type == 'Classification':
        if target_encoder is not None:
            decoded_preds = target_encoder.inverse_transform(raw_preds.astype(int))
            result_df[pred_col_name] = decoded_preds
            classes = list(target_encoder.classes_)
        else:
            result_df[pred_col_name] = raw_preds
            classes = [str(c) for c in getattr(model, 'classes_', [0, 1])]

        # Add probabilities if supported
        if hasattr(model, 'predict_proba'):
            try:
                probs = model.predict_proba(X_proc)
                max_confidence = np.max(probs, axis=1) * 100
                result_df['Confidence (%)'] = np.round(max_confidence, 2)

                for idx, cls in enumerate(classes):
                    result_df[f"Prob_{cls} (%)"] = np.round(probs[:, idx] * 100, 2)
            except Exception:
                pass
    else:
        result_df[pred_col_name] = np.round(raw_preds, 4)

    return result_df, None
