"""
Model Comparison Module for Predictive Modeling Dashboard.
Builds comparison tables, interactive multi-metric comparison charts,
and automated algorithmic performance explanations.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import plotly.express as px
import plotly.graph_objects as go


def build_comparison_dataframe(results: Dict[str, Any], problem_type: str) -> pd.DataFrame:
    """
    Construct a clean, ranked comparison table of all trained algorithms.
    """
    rows = []
    for name, data in results.items():
        if 'error' in data or 'metrics' not in data:
            continue
        m = data['metrics']
        row = {
            'Algorithm': name,
            'Model Family': data.get('type', 'N/A'),
            'Training Time (s)': data.get('train_time_sec', 0.0)
        }
        if problem_type == 'Classification':
            row['Accuracy'] = m.get('Accuracy', 0.0)
            row['Precision'] = m.get('Precision', 0.0)
            row['Recall'] = m.get('Recall', 0.0)
            row['F1 Score'] = m.get('F1 Score', 0.0)
            row['AUC Score'] = m.get('AUC Score', 'N/A')
        else:
            row['R2 Score'] = m.get('R2 Score', 0.0)
            row['RMSE'] = m.get('RMSE', 0.0)
            row['MAE'] = m.get('MAE', 0.0)
            row['MSE'] = m.get('MSE', 0.0)

        rows.append(row)

    df_comp = pd.DataFrame(rows)
    if df_comp.empty:
        return df_comp

    # Sort descending by primary metric
    if problem_type == 'Classification':
        df_comp = df_comp.sort_values(by=['F1 Score', 'Accuracy'], ascending=[False, False]).reset_index(drop=True)
    else:
        df_comp = df_comp.sort_values(by=['R2 Score', 'RMSE'], ascending=[False, True]).reset_index(drop=True)

    # Insert Rank column
    df_comp.insert(0, 'Rank', [f"#{i+1}" for i in range(len(df_comp))])
    return df_comp


def plot_model_comparison_bar(df_comp: pd.DataFrame, problem_type: str) -> go.Figure:
    """
    Interactive Grouped Bar Chart comparing models across key metrics.
    """
    if df_comp.empty:
        return go.Figure()

    if problem_type == 'Classification':
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        palette = ['#3B82F6', '#10B981', '#F59E0B', '#8B5CF6']
    else:
        # Normalize/display R2 Score and relative MAE
        metrics = ['R2 Score']
        palette = ['#2563EB']

    fig = go.Figure()
    for idx, metric in enumerate(metrics):
        if metric in df_comp.columns:
            fig.add_trace(go.Bar(
                x=df_comp['Algorithm'],
                y=df_comp[metric],
                name=metric,
                marker_color=palette[idx % len(palette)],
                text=df_comp[metric].apply(lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else str(x)),
                textposition='auto'
            ))

    fig.update_layout(
        barmode='group',
        title=dict(text="Model Performance Comparison by Metric", font=dict(size=18, color="#1E3A8A")),
        xaxis_title="Algorithm",
        yaxis_title="Score",
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=450,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#F8FAFC'
    )
    return fig


def plot_radar_comparison(df_comp: pd.DataFrame, problem_type: str) -> Optional[go.Figure]:
    """
    Spider / Radar chart comparing top models across multi-dimensional metrics (Classification).
    """
    if problem_type != 'Classification' or df_comp.empty:
        return None

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    categories = metrics + [metrics[0]] # Close the loop

    fig = go.Figure()
    colors = ['#2563EB', '#7C3AED', '#059669', '#D97706', '#DC2626']

    for i, row in df_comp.iterrows():
        values = [row[m] for m in metrics]
        values.append(values[0])
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=row['Algorithm'],
            line=dict(color=colors[i % len(colors)], width=2),
            opacity=0.6
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1.0])
        ),
        showlegend=True,
        title=dict(text="Multivariate Performance Radar", font=dict(size=18, color="#1E3A8A")),
        height=450,
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def explain_best_model(
    best_model_name: str,
    df_comp: pd.DataFrame,
    problem_type: str
) -> str:
    """
    Generate an intelligent narrative explaining why the winning algorithm outperformed
    the others, referencing specific metrics and algorithmic inductive biases.
    """
    if df_comp.empty or best_model_name not in df_comp['Algorithm'].values:
        return "Model comparison details are unavailable."

    best_row = df_comp[df_comp['Algorithm'] == best_model_name].iloc[0]
    runner_up = df_comp.iloc[1] if len(df_comp) > 1 else None

    # Algorithmic characteristic blurbs
    algo_strengths = {
        "Random Forest": "its ensemble bagging mechanism, which aggregates predictions across 100 decorrelated decision trees, dramatically suppressing model variance and preventing overfitting to noisy training patterns.",
        "Random Forest Regressor": "its ensemble averaging across hundreds of decision trees, dampening individual leaf variance while capturing non-linear interactions across high-dimensional features.",
        "Decision Tree": "its ability to create direct orthogonal splits across critical thresholds without assuming linearity, effectively isolating high-density decision regions.",
        "Decision Tree Regressor": "its non-linear recursive partitioning of the feature space, finding localized mean targets directly.",
        "Logistic Regression": "the well-calibrated linear separability of the feature representations, maximizing log-likelihood while penalizing extreme weight values via L2 regularization.",
        "Linear Regression": "the strong linear correlation present between independent features and the target variable, minimizing the total squared residuals without overparameterization.",
        "Support Vector Machine (SVM)": "the Radial Basis Function (RBF) kernel mapping inputs into infinite-dimensional Hilbert space, finding the maximum-margin hyperplane with high generalization power.",
        "Support Vector Regressor (SVR)": "the epsilon-insensitive loss function, ignoring errors below a specified tolerance and concentrating solely on support vectors to build a robust boundary.",
        "K-Nearest Neighbors (KNN)": "local manifold smoothness, allowing instance-based non-parametric classification directly from neighborhood feature similarity.",
        "K-Nearest Neighbors Regressor": "local distance-weighted interpolation, predicting smooth continuous outcomes directly from neighboring observations."
    }

    strength_text = algo_strengths.get(
        best_model_name,
        "its superior balance between bias and variance across the evaluation split."
    )

    if problem_type == 'Classification':
        f1 = best_row.get('F1 Score', 0.0)
        acc = best_row.get('Accuracy', 0.0)
        prec = best_row.get('Precision', 0.0)
        rec = best_row.get('Recall', 0.0)

        explanation = f"""
### 🏆 Why **{best_model_name}** Ranked #1:

1. **Superior Overall Balance**: Achieved the highest **F1 Score of {f1:.4f}** and an **Accuracy of {acc * 100:.2f}%**, demonstrating harmonious balance between Precision ({prec:.4f}) and Recall ({rec:.4f}).
2. **Algorithmic Advantage**: **{best_model_name}** excelled due to {strength_text}
"""
        if runner_up is not None:
            r_name = runner_up['Algorithm']
            r_f1 = runner_up.get('F1 Score', 0.0)
            diff = (f1 - r_f1) * 100
            explanation += f"3. **Head-to-Head Comparison**: Outperformed the runner-up **{r_name}** (F1: {r_f1:.4f}) by **+{diff:.2f}% percentage points**, demonstrating superior generalization on unseen test samples."

    else:
        r2 = best_row.get('R2 Score', 0.0)
        rmse = best_row.get('RMSE', 0.0)
        mae = best_row.get('MAE', 0.0)

        explanation = f"""
### 🏆 Why **{best_model_name}** Ranked #1:

1. **Superior Predictive Accuracy**: Achieved the highest **$R^2$ Score of {r2:.4f}**, explaining approximately **{max(0.0, r2)*100:.2f}%** of the total variance in the target variable.
2. **Lowest Error Rates**: Minimized prediction discrepancy with an **RMSE of {rmse:.4f}** and an average **MAE of {mae:.4f}**.
3. **Algorithmic Advantage**: **{best_model_name}** achieved the lowest loss because of {strength_text}
"""
        if runner_up is not None:
            r_name = runner_up['Algorithm']
            r_r2 = runner_up.get('R2 Score', 0.0)
            explanation += f"4. **Head-to-Head Comparison**: Outperformed the runner-up **{r_name}** ($R^2$: {r_r2:.4f}) with higher explained variance and tighter error bounds."

    return explanation
