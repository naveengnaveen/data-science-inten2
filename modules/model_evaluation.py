"""
Model Evaluation Module for Predictive Modeling Dashboard.
Computes evaluation metrics (Classification & Regression) and generates
interactive Plotly visualizations as well as Matplotlib/Seaborn figures.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import matplotlib
matplotlib.use('Agg') # Headless backend for web app
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==============================================================================
# Metric Calculations
# ==============================================================================

def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    target_classes: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Calculate full suite of classification metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    n_classes = len(np.unique(y_true))
    is_binary = (n_classes == 2)
    avg_type = 'binary' if is_binary else 'weighted'

    prec = precision_score(y_true, y_pred, average=avg_type, zero_division=0)
    rec = recall_score(y_true, y_pred, average=avg_type, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=avg_type, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # ROC AUC
    auc_score = None
    roc_data = None
    if y_prob is not None:
        try:
            if is_binary:
                # Use positive class probabilities
                prob_pos = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
                auc_score = roc_auc_score(y_true, prob_pos)
                fpr, tpr, thresholds = roc_curve(y_true, prob_pos)
                roc_data = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': float(auc_score)}
            else:
                # Multiclass ROC AUC (One-vs-Rest)
                auc_score = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
                roc_data = {'auc': float(auc_score)}
        except Exception:
            auc_score = None

    labels = [str(c) for c in (target_classes if target_classes is not None else sorted(np.unique(y_true)))]

    return {
        'Accuracy': round(float(acc), 4),
        'Precision': round(float(prec), 4),
        'Recall': round(float(rec), 4),
        'F1 Score': round(float(f1), 4),
        'AUC Score': round(float(auc_score), 4) if auc_score is not None else 'N/A',
        'confusion_matrix': cm.tolist(),
        'class_labels': labels,
        'roc_data': roc_data
    }


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Calculate full suite of regression metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {
        'MAE': round(float(mae), 4),
        'MSE': round(float(mse), 4),
        'RMSE': round(float(rmse), 4),
        'R2 Score': round(float(r2), 4)
    }


# ==============================================================================
# Interactive Plotly Visualizations (Responsive & Styled)
# ==============================================================================

def plot_confusion_matrix_plotly(cm_array: List[List[int]], class_labels: List[str]) -> go.Figure:
    """
    Generate an elegant, interactive Confusion Matrix heatmap using Plotly.
    """
    z = np.array(cm_array)
    annotations = []
    for i in range(len(class_labels)):
        for j in range(len(class_labels)):
            val = z[i][j]
            annotations.append(
                dict(
                    x=class_labels[j],
                    y=class_labels[i],
                    text=f"<b>{val}</b>",
                    font=dict(color="white" if val > z.max()/2 else "#1E293B", size=14),
                    showarrow=False
                )
            )

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=class_labels,
        y=class_labels,
        colorscale=[[0, '#EFF6FF'], [0.5, '#60A5FA'], [1.0, '#1E3A8A']],
        showscale=True,
        hoverongaps=False
    ))

    fig.update_layout(
        title=dict(text="Confusion Matrix", font=dict(size=18, color="#1E3A8A", family="Arial")),
        xaxis=dict(title="Predicted Label", side="bottom"),
        yaxis=dict(title="Actual Label", autorange="reversed"),
        annotations=annotations,
        margin=dict(l=60, r=40, t=60, b=60),
        height=450,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def plot_roc_curve_plotly(roc_data: Optional[Dict[str, Any]], model_name: str = "Model") -> Optional[go.Figure]:
    """
    Generate an interactive ROC Curve with Plotly.
    """
    if not roc_data or 'fpr' not in roc_data or 'tpr' not in roc_data:
        return None

    fpr = roc_data['fpr']
    tpr = roc_data['tpr']
    auc_val = roc_data.get('auc', 0.0)

    fig = go.Figure()
    # ROC curve
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f"{model_name} (AUC = {auc_val:.3f})",
        line=dict(color='#2563EB', width=3)
    ))
    # Random chance line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Chance (AUC = 0.500)',
        line=dict(color='#94A3B8', width=2, dash='dash')
    ))

    fig.update_layout(
        title=dict(text=f"ROC Curve - {model_name}", font=dict(size=18, color="#1E3A8A")),
        xaxis=dict(title="False Positive Rate (1 - Specificity)", range=[-0.02, 1.02]),
        yaxis=dict(title="True Positive Rate (Sensitivity)", range=[-0.02, 1.05]),
        legend=dict(x=0.55, y=0.1, bgcolor='rgba(255,255,255,0.85)'),
        height=450,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#F8FAFC'
    )
    return fig


def plot_actual_vs_predicted_plotly(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> go.Figure:
    """
    Generate Actual vs Predicted scatter plot with ideal 45-degree reference line.
    """
    df_plot = pd.DataFrame({'Actual': y_true, 'Predicted': y_pred})
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())

    fig = go.Figure()
    # Scatter points
    fig.add_trace(go.Scatter(
        x=df_plot['Actual'],
        y=df_plot['Predicted'],
        mode='markers',
        marker=dict(color='#3B82F6', size=7, opacity=0.65, line=dict(color='#1D4ED8', width=0.5)),
        name='Predictions'
    ))
    # Ideal diagonal line
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Fit (y = x)',
        line=dict(color='#EF4444', width=2.5, dash='dash')
    ))

    fig.update_layout(
        title=dict(text=f"Actual vs Predicted Values - {model_name}", font=dict(size=18, color="#1E3A8A")),
        xaxis=dict(title="Actual Values"),
        yaxis=dict(title="Predicted Values"),
        legend=dict(x=0.05, y=0.95),
        height=450,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#F8FAFC'
    )
    return fig


def plot_prediction_distribution_plotly(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    problem_type: str,
    class_labels: Optional[List[str]] = None
) -> go.Figure:
    """
    Generate prediction distribution chart:
    - Classification: grouped bar chart comparing actual vs predicted class frequencies.
    - Regression: residual distribution (histogram + KDE).
    """
    if problem_type == 'Classification':
        labels = class_labels if class_labels else sorted(list(set(y_true).union(set(y_pred))))
        actual_counts = [int(np.sum(y_true == i)) for i in range(len(labels))]
        pred_counts = [int(np.sum(y_pred == i)) for i in range(len(labels))]

        fig = go.Figure(data=[
            go.Bar(name='Actual', x=labels, y=actual_counts, marker_color='#3B82F6'),
            go.Bar(name='Predicted', x=labels, y=pred_counts, marker_color='#8B5CF6')
        ])
        fig.update_layout(
            barmode='group',
            title=dict(text="Class Distribution: Actual vs Predicted", font=dict(size=18, color="#1E3A8A")),
            xaxis_title="Class",
            yaxis_title="Count",
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='#F8FAFC'
        )
        return fig
    else:
        # Residuals
        residuals = y_true - y_pred
        fig = go.Figure(data=[go.Histogram(
            x=residuals,
            nbinsx=30,
            marker_color='#6366F1',
            opacity=0.75,
            name='Residuals'
        )])
        fig.add_vline(x=0, line_dash='dash', line_color='#EF4444', line_width=2)
        fig.update_layout(
            title=dict(text="Residuals Distribution (Actual - Predicted)", font=dict(size=18, color="#1E3A8A")),
            xaxis_title="Residual Error",
            yaxis_title="Frequency",
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='#F8FAFC'
        )
        return fig


def plot_feature_importance_plotly(
    feature_names: List[str],
    importances: np.ndarray,
    top_n: int = 15,
    title: str = "Feature Importance"
) -> go.Figure:
    """
    Generate horizontal bar chart for top N feature importances.
    """
    # Absolute values in case of linear regression / logistic regression coefficients
    abs_imp = np.abs(importances)
    df_imp = pd.DataFrame({
        'Feature': feature_names[:len(abs_imp)],
        'Importance': abs_imp
    }).sort_values(by='Importance', ascending=True).tail(top_n)

    fig = go.Figure(go.Bar(
        x=df_imp['Importance'],
        y=df_imp['Feature'],
        orientation='h',
        marker=dict(
            color=df_imp['Importance'],
            colorscale=[[0, '#93C5FD'], [1, '#1D4ED8']]
        )
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color="#1E3A8A")),
        xaxis_title="Relative Importance / Magnitude",
        yaxis_title="Feature",
        height=max(400, top_n * 26),
        margin=dict(l=150, r=40, t=50, b=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#F8FAFC'
    )
    return fig


def plot_correlation_heatmap_plotly(df: pd.DataFrame, max_cols: int = 15) -> Optional[go.Figure]:
    """
    Interactive Correlation Heatmap for numerical columns.
    """
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return None

    # Limit to top columns by variance if too large
    if num_df.shape[1] > max_cols:
        top_cols = num_df.var().sort_values(ascending=False).head(max_cols).index
        num_df = num_df[top_cols]

    corr = num_df.corr().round(2)
    cols = list(corr.columns)

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=cols,
        y=cols,
        colorscale='RdBu_r',
        zmin=-1, zmax=1,
        hoverongaps=False
    ))

    fig.update_layout(
        title=dict(text="Feature Correlation Heatmap", font=dict(size=18, color="#1E3A8A")),
        height=550,
        margin=dict(l=80, r=40, t=60, b=80),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig
