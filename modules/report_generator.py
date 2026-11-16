"""
Report Generator Module for Predictive Modeling Dashboard.
Generates comprehensive PDF reports using ReportLab, plus CSV/JSON exports
for model comparisons, evaluation metrics, and prediction batches.
"""

import io
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    KeepTogether
)


def generate_pdf_report(
    dataset_name: str,
    target_col: str,
    problem_type: str,
    df_comp: pd.DataFrame,
    best_model_name: str,
    dataset_summary: Dict[str, Any],
    preprocessing_summary: Dict[str, Any],
    model_explanation: str
) -> bytes:
    """
    Generate an executive PDF Project Report with tables, KPI summaries,
    and algorithmic rationales. Returns raw PDF bytes for Streamlit download.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    
    # Custom palette
    primary_color = colors.HexColor('#1E3A8A')   # Navy / Blue
    accent_color = colors.HexColor('#6366F1')    # Purple
    text_dark = colors.HexColor('#1E293B')       # Slate dark
    bg_light = colors.HexColor('#F8FAFC')        # Soft background
    border_color = colors.HexColor('#E2E8F0')

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=primary_color,
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#64748B'),
        spaceAfter=15
    )

    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=primary_color,
        spaceBefore=12,
        spaceAfter=8
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=text_dark,
        spaceAfter=6
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.white
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=text_dark
    )

    story = []

    # 1. Header Banner
    story.append(Paragraph("Predictive Modeling Using Machine Learning", title_style))
    now_str = datetime.now().strftime("%B %d, %Y - %H:%M:%S")
    story.append(Paragraph(f"Automated Machine Learning Performance & Evaluation Audit | Generated on {now_str}", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=primary_color, spaceAfter=14))

    # 2. Executive Summary Box
    story.append(Paragraph("1. Executive Summary", h1_style))
    summary_text = (
        f"This report synthesizes the empirical results obtained from training supervised machine learning algorithms "
        f"on the <b>{dataset_name}</b> dataset. The system dynamically identified the target variable <b>'{target_col}'</b> "
        f"as a <b>{problem_type}</b> problem. All feature engineering and scaling transformations were conducted strictly "
        f"within Scikit-learn validation pipelines to eliminate data leakage. The top-performing algorithm selected is "
        f"<b>{best_model_name}</b>."
    )
    story.append(Paragraph(summary_text, body_style))
    story.append(Spacer(1, 8))

    # 3. Key Project Metrics KPI Table
    kpi_data = [
        [
            Paragraph("<b>Total Records</b>", body_style),
            Paragraph(str(dataset_summary.get('num_rows', 'N/A')), body_style),
            Paragraph("<b>Target Column</b>", body_style),
            Paragraph(str(target_col), body_style)
        ],
        [
            Paragraph("<b>Features Used</b>", body_style),
            Paragraph(str(len(preprocessing_summary.get('feature_cols', []))), body_style),
            Paragraph("<b>Problem Type</b>", body_style),
            Paragraph(str(problem_type), body_style)
        ],
        [
            Paragraph("<b>Train / Test Split</b>", body_style),
            Paragraph(f"{int((1 - preprocessing_summary.get('test_size', 0.2))*100)}% / {int(preprocessing_summary.get('test_size', 0.2)*100)}%", body_style),
            Paragraph("<b>Winning Model</b>", body_style),
            Paragraph(f"<b>{best_model_name}</b>", body_style)
        ]
    ]

    kpi_table = Table(kpi_data, colWidths=[1.8*inch, 1.4*inch, 1.8*inch, 2.0*inch])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), bg_light),
        ('BOX', (0, 0), (-1, -1), 1, border_color),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, border_color),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 14))

    # 4. Data Preprocessing Pipeline Audit
    story.append(Paragraph("2. Data Preprocessing & Leakage Control", h1_style))
    prep_text = (
        f"• <b>Duplicates Removed:</b> {preprocessing_summary.get('duplicates_removed', 0)} rows<br/>"
        f"• <b>Imputation:</b> Median for numerical features, Mode ('most_frequent') for categorical<br/>"
        f"• <b>Categorical Encoding:</b> {preprocessing_summary.get('encoding_method', 'One-Hot Encoding')}<br/>"
        f"• <b>Numerical Scaling:</b> {preprocessing_summary.get('scaling_method', 'StandardScaler')}<br/>"
        f"• <b>Outlier Treatment:</b> Interquartile Range (IQR) boundary clipping applied on training split."
    )
    story.append(Paragraph(prep_text, body_style))
    story.append(Spacer(1, 12))

    # 5. Model Comparison Table
    story.append(Paragraph("3. Algorithmic Benchmark & Performance Comparison", h1_style))
    if not df_comp.empty:
        # Prepare table headers and rows
        display_cols = [c for c in df_comp.columns if c not in ['Model Family']]
        headers = [Paragraph(f"<b>{c}</b>", table_header_style) for c in display_cols]
        t_data = [headers]

        for _, row in df_comp.iterrows():
            row_cells = []
            for c in display_cols:
                val = row[c]
                val_str = f"{val:.4f}" if isinstance(val, (float, np.floating)) else str(val)
                row_cells.append(Paragraph(val_str, table_cell_style))
            t_data.append(row_cells)

        col_w = (7.0 * inch) / len(display_cols)
        comp_table = Table(t_data, colWidths=[col_w] * len(display_cols))
        comp_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), primary_color),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, border_color),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, bg_light]),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(comp_table)
    else:
        story.append(Paragraph("No trained models recorded.", body_style))

    story.append(Spacer(1, 14))

    # 6. Winning Model Insights
    story.append(Paragraph("4. Architectural Rationale & Recommendations", h1_style))
    clean_explanation = model_explanation.replace('#', '').replace('**', '')
    for para in clean_explanation.split('\n\n'):
        if para.strip():
            story.append(Paragraph(para.strip(), body_style))

    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=0.75, color=border_color, spaceAfter=8))
    story.append(Paragraph("End of Machine Learning Evaluation Audit | Generated via Predictive Modeling ML Platform", subtitle_style))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
