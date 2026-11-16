"""
Predictive Modeling Using Machine Learning
A production-grade Streamlit Dashboard for End-to-End Supervised ML:
- Dataset ingestion & automated problem type detection
- Leak-free preprocessing pipelines
- Supervised classification & regression training (5 algorithms each)
- Comprehensive evaluation & interactive Plotly diagnostics
- Model comparison & algorithmic explanations
- Real-time single & bulk batch prediction
- Automated PDF report & Joblib export
"""

import os
import sys
import io
import json
import time
import pandas as pd
import numpy as np
import streamlit as st

# Silence joblib physical core warning on Windows
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# Set up module search path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from modules.data_loader import (
    load_dataset,
    get_dataset_overview,
    detect_id_columns,
    detect_problem_type
)
from modules.preprocessing import (
    preprocess_and_split
)
from modules.classification_models import get_classification_models
from modules.regression_models import get_regression_models
from modules.model_training import (
    train_multiple_models,
    identify_best_model,
    save_trained_pipeline,
    load_trained_pipeline
)
from modules.model_evaluation import (
    plot_confusion_matrix_plotly,
    plot_roc_curve_plotly,
    plot_actual_vs_predicted_plotly,
    plot_prediction_distribution_plotly,
    plot_feature_importance_plotly,
    plot_correlation_heatmap_plotly
)
from modules.model_comparison import (
    build_comparison_dataframe,
    plot_model_comparison_bar,
    plot_radar_comparison,
    explain_best_model
)
from modules.prediction import (
    predict_single_record,
    predict_batch
)
from modules.report_generator import (
    generate_pdf_report
)

# ==============================================================================
# Page Configuration & Modern Light Theme CSS
# ==============================================================================

st.set_page_config(
    page_title="Predictive Modeling ML",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Background & Main Container */
    .main {
        background-color: #F8FAFC;
    }

    /* Headings */
    h1, h2, h3, h4 {
        color: #1E3A8A;
        font-weight: 700;
    }

    /* Modern KPI Cards */
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        position: relative;
        overflow: hidden;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }
    .kpi-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #2563EB, #7C3AED);
    }
    .kpi-title {
        color: #64748B;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .kpi-value {
        color: #1E293B;
        font-size: 1.85rem;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .kpi-subtitle {
        color: #10B981;
        font-size: 0.8rem;
        font-weight: 500;
    }

    /* Info & Alert Badges */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-blue {
        background-color: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #BFDBFE;
    }
    .badge-purple {
        background-color: #F5F3FF;
        color: #6D28D9;
        border: 1px solid #DDD6FE;
    }
    .badge-green {
        background-color: #ECFDF5;
        color: #047857;
        border: 1px solid #A7F3D0;
    }

    /* Custom Streamlit Button Styling */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        color: #FFFFFF;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        box-shadow: 0 2px 4px rgba(37, 99, 235, 0.2);
        transition: all 0.2s ease;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #1D4ED8 0%, #1E3A8A 100%);
        box-shadow: 0 4px 8px rgba(37, 99, 235, 0.35);
        transform: translateY(-1px);
    }

    /* Prediction Result Banner */
    .prediction-box {
        background: linear-gradient(135deg, #EFF6FF 0%, #F5F3FF 100%);
        border: 2px solid #818CF8;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    .prediction-box h2 {
        color: #1E3A8A;
        font-size: 2.2rem;
        margin: 8px 0;
    }

    /* Sidebar Clean Styling */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# Session State Initialization
# ==============================================================================

def init_session_state():
    defaults = {
        'df_raw': None,
        'dataset_name': 'No dataset loaded',
        'target_col': None,
        'problem_type': 'Classification',
        'feature_cols': [],
        'id_cols': [],
        'preprocessed_data': None,
        'training_results': None,
        'best_model_name': None,
        'recent_predictions': [],
        'saved_model_path': None
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()


# ==============================================================================
# Helper: Quick Sample Dataset Loader
# ==============================================================================

def load_predefined_sample(sample_type: str = 'classification'):
    sample_dir = os.path.join(CURRENT_DIR, 'dataset')
    if sample_type == 'classification':
        path = os.path.join(sample_dir, 'sample_dataset.csv')
        name = 'Customer Churn (Classification Sample)'
        target = 'Churn'
    else:
        path = os.path.join(sample_dir, 'sample_housing_dataset.csv')
        name = 'Housing Prices (Regression Sample)'
        target = 'HousePrice'

    if os.path.exists(path):
        df = pd.read_csv(path)
        st.session_state['df_raw'] = df
        st.session_state['dataset_name'] = name
        st.session_state['target_col'] = target
        
        # Reset downstream states
        prob_info = detect_problem_type(df, target)
        st.session_state['problem_type'] = prob_info['problem_type']
        id_cols = detect_id_columns(df)
        st.session_state['id_cols'] = id_cols
        st.session_state['feature_cols'] = [c for c in df.columns if c != target and c not in id_cols]
        st.session_state['preprocessed_data'] = None
        st.session_state['training_results'] = None
        st.session_state['best_model_name'] = None


# ==============================================================================
# Sidebar Navigation & System Status
# ==============================================================================

with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding: 10px 0 15px 0;">
            <h2 style="color: #1E3A8A; margin: 0; font-size: 1.4rem;">⚡ AutoPredict ML</h2>
            <p style="color: #64748B; font-size: 0.8rem; margin: 2px 0 0 0;">Intelligent Supervised Machine Learning</p>
        </div>
    """, unsafe_allow_html=True)

    pages = [
        "1. 📊 Dashboard Overview",
        "2. 📁 Upload Dataset",
        "3. ⚙️ Data Preprocessing",
        "4. 🚀 Model Training",
        "5. 📈 Model Evaluation",
        "6. ⚖️ Model Comparison",
        "7. 🔮 Make Predictions",
        "8. 📥 Download Reports"
    ]

    selected_page = st.radio("Navigate", pages, label_visibility="collapsed")

    st.markdown("---")
    st.markdown("### 📌 Workspace Status")

    if st.session_state['df_raw'] is not None:
        st.markdown(f"**Dataset:** `{st.session_state['dataset_name'][:22]}`")
        st.markdown(f"**Shape:** `{st.session_state['df_raw'].shape[0]} rows × {st.session_state['df_raw'].shape[1]} cols`")
        if st.session_state['target_col']:
            st.markdown(f"**Target:** `{st.session_state['target_col']}`")
            prob_badge = "badge-blue" if st.session_state['problem_type'] == 'Classification' else "badge-purple"
            st.markdown(f"**Task:** <span class='status-badge {prob_badge}'>{st.session_state['problem_type']}</span>", unsafe_allow_html=True)

        if st.session_state['preprocessed_data'] is not None:
            st.markdown("<span class='status-badge badge-green'>✓ Data Preprocessed</span>", unsafe_allow_html=True)
        if st.session_state['training_results'] is not None:
            st.markdown(f"<span class='status-badge badge-green'>✓ {len(st.session_state['training_results'])} Models Trained</span>", unsafe_allow_html=True)
            if st.session_state['best_model_name']:
                st.markdown(f"**Best:** 🏆 `{st.session_state['best_model_name']}`")
    else:
        st.info("No dataset loaded yet. Please upload a file or load a sample.")

    st.markdown("---")
    st.markdown("### 🚀 Quick Demo Datasets")
    col_demo1, col_demo2 = st.columns(2)
    with col_demo1:
        if st.button("👥 Churn", help="Load Classification Sample Dataset"):
            load_predefined_sample('classification')
            st.rerun()
    with col_demo2:
        if st.button("🏡 Housing", help="Load Regression Sample Dataset"):
            load_predefined_sample('regression')
            st.rerun()


# ==============================================================================
# PAGE 1: DASHBOARD OVERVIEW
# ==============================================================================

if "Dashboard Overview" in selected_page:
    st.title("📊 Predictive Modeling Dashboard")
    st.markdown("Monitor high-level dataset statistics, model benchmark rankings, and real-time prediction activity.")

    df = st.session_state['df_raw']
    prep = st.session_state['preprocessed_data']
    res = st.session_state['training_results']
    best_name = st.session_state['best_model_name']
    prob_type = st.session_state['problem_type']

    # --- KPI Cards Row ---
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        n_rows = df.shape[0] if df is not None else 0
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Total Records</div>
                <div class="kpi-value">{n_rows:,}</div>
                <div class="kpi-subtitle">Rows in Dataset</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        n_feats = len(st.session_state['feature_cols']) if df is not None else 0
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Features</div>
                <div class="kpi-value">{n_feats}</div>
                <div class="kpi-subtitle">Selected Predictors</div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        n_trained = len(res) if res is not None else 0
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Models Trained</div>
                <div class="kpi-value">{n_trained}</div>
                <div class="kpi-subtitle">Algorithms Evaluated</div>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        display_best = best_name if best_name else "None"
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Best Model</div>
                <div class="kpi-value" style="font-size: 1.25rem;">{display_best}</div>
                <div class="kpi-subtitle">Top Performer</div>
            </div>
        """, unsafe_allow_html=True)

    with col5:
        if res and best_name and best_name in res and 'metrics' in res[best_name]:
            m = res[best_name]['metrics']
            if prob_type == 'Classification':
                perf_str = f"{m.get('F1 Score', 0.0):.3f} F1"
                sub_perf = f"Acc: {m.get('Accuracy', 0.0)*100:.1f}%"
            else:
                perf_str = f"{m.get('R2 Score', 0.0):.3f} R²"
                sub_perf = f"RMSE: {m.get('RMSE', 0.0):,.1f}"
        else:
            perf_str = "N/A"
            sub_perf = "Awaiting Training"

        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Top Metric</div>
                <div class="kpi-value">{perf_str}</div>
                <div class="kpi-subtitle">{sub_perf}</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # --- Training & Preprocessing Summary ---
    if df is not None:
        c_left, c_right = st.columns([3, 2])
        with c_left:
            st.subheader("📋 Dataset Preview")
            st.dataframe(df.head(6), use_container_width=True)

        with c_right:
            st.subheader("⚙️ Preprocessing & Split Information")
            if prep is not None:
                st.markdown(f"""
                - **Target Variable:** `{st.session_state['target_col']}` ({prob_type})
                - **Train Samples:** `{prep['train_samples']:,}` ({int((1-prep['test_size'])*100)}%)
                - **Test Samples:** `{prep['test_samples']:,}` ({int(prep['test_size']*100)}%)
                - **Numerical Scaler:** `{prep['scaling_method']}`
                - **Categorical Encoder:** `{prep['encoding_method']}`
                - **Duplicates Removed:** `{prep['duplicates_removed']}`
                """)
            else:
                st.info("Dataset loaded. Proceed to **⚙️ Data Preprocessing** to configure pipelines.")

        # --- Model Performance Overview Chart ---
        if res is not None:
            st.subheader("📈 Algorithmic Leaderboard")
            comp_df = build_comparison_dataframe(res, prob_type)
            fig_bar = plot_model_comparison_bar(comp_df, prob_type)
            st.plotly_chart(fig_bar, use_container_width=True)

        # --- Recent Predictions ---
        if st.session_state['recent_predictions']:
            st.subheader("🔮 Recent Predictions Log")
            st.dataframe(pd.DataFrame(st.session_state['recent_predictions']), use_container_width=True)

    else:
        st.info("👋 Welcome! To begin, click **📁 Upload Dataset** in the sidebar or test one of the sample datasets (Churn / Housing).")


# ==============================================================================
# PAGE 2: UPLOAD DATASET
# ==============================================================================

elif "Upload Dataset" in selected_page:
    st.title("📁 Dataset Upload & Inspection")
    st.markdown("Upload any CSV or Excel dataset. The platform automatically inspects columns, data types, missing records, and infers the machine learning task.")

    # Upload widget
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload tabular data with headers."
    )

    if uploaded_file is not None:
        try:
            df = load_dataset(uploaded_file)
            st.session_state['df_raw'] = df
            st.session_state['dataset_name'] = uploaded_file.name
            st.success(f"Successfully loaded '{uploaded_file.name}' ({len(df):,} rows, {len(df.columns)} columns)!")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

    df = st.session_state['df_raw']

    if df is not None:
        overview = get_dataset_overview(df)

        st.markdown("---")
        # Overview Metrics Row
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Rows", f"{overview['num_rows']:,}")
        m2.metric("Columns", overview['num_columns'])
        m3.metric("Missing Cells", f"{overview['total_missing']:,}", f"{overview['missing_percentage']}%")
        m4.metric("Duplicate Rows", f"{overview['duplicate_count']:,}")
        m5.metric("RAM Footprint", f"{overview['memory_mb']} MB")

        # Preview Data Tabs
        st.subheader("👀 Dataset Inspection")
        tab_head, tab_tail, tab_schema, tab_stats = st.tabs([
            "First 5 Records (Head)",
            "Last 5 Records (Tail)",
            "Columns & Data Types",
            "Numerical Statistics"
        ])

        with tab_head:
            st.dataframe(df.head(5), use_container_width=True)

        with tab_tail:
            st.dataframe(df.tail(5), use_container_width=True)

        with tab_schema:
            st.dataframe(overview['columns_summary'], use_container_width=True)

        with tab_stats:
            num_df = df.select_dtypes(include=[np.number])
            if not num_df.empty:
                st.dataframe(num_df.describe().T, use_container_width=True)
            else:
                st.info("No numerical columns found.")

        # Target Column & Problem Type Configuration
        st.markdown("---")
        st.subheader("🎯 Target Column & Problem Type Selection")
        col_t1, col_t2 = st.columns([1, 1])

        with col_t1:
            all_cols = list(df.columns)
            curr_target = st.session_state['target_col']
            default_idx = all_cols.index(curr_target) if (curr_target in all_cols) else (len(all_cols) - 1)
            target_selection = st.selectbox(
                "Select Target Column (Dependent Variable)",
                all_cols,
                index=default_idx
            )
            st.session_state['target_col'] = target_selection

        # Dynamic detection
        prob_detection = detect_problem_type(df, target_selection)

        with col_t2:
            st.markdown("**Automated Inference:**")
            st.info(prob_detection['reason'])

            override_type = st.radio(
                "Confirm or Override Problem Type:",
                ["Classification", "Regression"],
                index=0 if prob_detection['problem_type'] == 'Classification' else 1,
                horizontal=True
            )
            st.session_state['problem_type'] = override_type

        # Feature Selection
        st.markdown("---")
        st.subheader("🧩 Feature Selection")
        id_candidates = detect_id_columns(df)
        if id_candidates:
            st.warning(f"Detected likely identifier columns (high uniqueness): `{', '.join(id_candidates)}`. These are excluded by default to avoid spurious correlations.")

        available_features = [c for c in df.columns if c != target_selection]
        default_selected_features = [c for c in available_features if c not in id_candidates]

        selected_features = st.multiselect(
            "Select Features for Model Training",
            options=available_features,
            default=default_selected_features
        )
        st.session_state['feature_cols'] = selected_features
        st.session_state['id_cols'] = id_candidates

        if not selected_features:
            st.error("Please select at least one feature column for predictive modeling.")
        else:
            st.success(f"{len(selected_features)} features selected for modeling.")


# ==============================================================================
# PAGE 3: DATA PREPROCESSING
# ==============================================================================

elif "Data Preprocessing" in selected_page:
    st.title("⚙️ Data Preprocessing & Validation Pipeline")
    st.markdown("Construct leak-free Scikit-Learn transformers to handle missing values, duplicates, outliers, scaling, and categorical encoding.")

    df = st.session_state['df_raw']
    target_col = st.session_state['target_col']
    features = st.session_state['feature_cols']
    problem_type = st.session_state['problem_type']

    if df is None or not target_col or not features:
        st.warning("Please upload a dataset and configure the target column first in **📁 Upload Dataset**.")
    else:
        # Preprocessing Options Form
        st.subheader("🛠️ Pipeline Configuration")
        c1, c2, c3 = st.columns(3)

        with c1:
            test_split = st.slider(
                "Test Split Ratio",
                min_value=0.10,
                max_value=0.40,
                value=0.20,
                step=0.05,
                help="Default is 80% train and 20% test."
            )
            remove_dups = st.checkbox("Remove Duplicate Records", value=True)

        with c2:
            num_imputer = st.selectbox(
                "Numerical Imputation",
                ["median", "mean", "most_frequent"],
                index=0
            )
            cat_imputer = st.selectbox(
                "Categorical Imputation",
                ["most_frequent", "constant"],
                index=0
            )

        with c3:
            scaler_opt = st.selectbox(
                "Feature Scaling",
                ["StandardScaler", "MinMaxScaler", "RobustScaler", "None"],
                index=0
            )
            encoding_opt = st.selectbox(
                "Categorical Encoding",
                ["One-Hot Encoding", "Ordinal Encoding"],
                index=0
            )
            handle_outliers = st.checkbox("Handle Outliers (IQR Clipping)", value=True)

        if st.button("🚀 Run Preprocessing Pipeline", use_container_width=True):
            with st.spinner("Executing leak-free transformation pipeline..."):
                try:
                    prep_dict = preprocess_and_split(
                        df=df,
                        target_col=target_col,
                        feature_cols=features,
                        problem_type=problem_type,
                        test_size=test_split,
                        random_state=42,
                        remove_duplicates=remove_dups,
                        impute_num_strategy=num_imputer,
                        impute_cat_strategy=cat_imputer,
                        scaling_method=scaler_opt,
                        encoding_method=encoding_opt,
                        handle_outliers=handle_outliers
                    )
                    st.session_state['preprocessed_data'] = prep_dict
                    # Invalidate stale training results when preprocessed data changes
                    st.session_state['training_results'] = None
                    st.session_state['best_model_name'] = None
                    st.success("Preprocessing completed successfully!")
                except Exception as e:
                    st.error(f"Preprocessing failed: {str(e)}")

        prep = st.session_state['preprocessed_data']
        if prep is not None:
            st.markdown("---")
            st.subheader("📊 Preprocessing Pipeline Results")

            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Training Samples", f"{prep['train_samples']:,}", f"{(1-prep['test_size'])*100:.0f}%")
            p2.metric("Testing Samples", f"{prep['test_samples']:,}", f"{prep['test_size']*100:.0f}%")
            p3.metric("Transformed Features", len(prep['feature_names_out']))
            p4.metric("Duplicates Dropped", prep['duplicates_removed'])

            # Outlier Audit Table
            if prep['outlier_stats']:
                st.markdown("#### 🔍 Outlier Detection Summary (IQR Method)")
                out_df = pd.DataFrame(prep['outlier_stats']).T
                out_df = out_df[['q25', 'q75', 'lower_bound', 'upper_bound', 'outlier_count', 'outlier_pct']]
                out_df.columns = ['Q1 (25%)', 'Q3 (75%)', 'Lower Bound', 'Upper Bound', 'Outlier Count', 'Outliers (%)']
                st.dataframe(out_df, use_container_width=True)

            # Transformed features preview
            st.markdown("#### 🔬 Processed Training Set Sample")
            preview_cols = prep['feature_names_out']
            preview_df = pd.DataFrame(prep['X_train'][:6], columns=preview_cols)
            st.dataframe(preview_df, use_container_width=True)


# ==============================================================================
# PAGE 4: MODEL TRAINING
# ==============================================================================

elif "Model Training" in selected_page:
    st.title("🚀 Supervised Model Training")
    st.markdown("Train multiple machine learning algorithms simultaneously, track training speed, and select the optimal model.")

    prep = st.session_state['preprocessed_data']
    prob_type = st.session_state['problem_type']

    if prep is None:
        st.warning("Please preprocess the dataset first in **⚙️ Data Preprocessing**.")
    else:
        st.markdown(f"**Task Type:** `{prob_type}` | **Target:** `{prep['target_col']}`")

        # Get candidates
        if prob_type == 'Classification':
            available_models = get_classification_models()
        else:
            available_models = get_regression_models()

        st.subheader("Select Algorithms to Train")
        col_algo1, col_algo2 = st.columns([3, 1])

        with col_algo1:
            selected_model_names = st.multiselect(
                "Algorithms",
                options=list(available_models.keys()),
                default=list(available_models.keys())
            )

        with col_algo2:
            st.markdown("<br/>", unsafe_allow_html=True)
            select_all = st.checkbox("Select All Algorithms", value=True)
            if not select_all:
                pass

        # Algorithm Details Expander
        with st.expander("ℹ️ Algorithm Details & Architectural Profiles"):
            for m_name, m_info in available_models.items():
                st.markdown(f"- **{m_name}** ({m_info['type']}): {m_info['description']}")

        if st.button("🏋️ Train Selected Models", use_container_width=True):
            if not selected_model_names:
                st.error("Please select at least one algorithm to train.")
            else:
                subset_dict = {k: available_models[k] for k in selected_model_names}
                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(pct, msg):
                    progress_bar.progress(pct)
                    status_text.text(msg)

                train_results = train_multiple_models(
                    models_dict=subset_dict,
                    X_train=prep['X_train'],
                    y_train=prep['y_train'],
                    X_test=prep['X_test'],
                    y_test=prep['y_test'],
                    problem_type=prob_type,
                    target_classes=prep['target_classes'],
                    progress_callback=update_progress
                )

                best_name = identify_best_model(train_results, prob_type)
                st.session_state['training_results'] = train_results
                st.session_state['best_model_name'] = best_name
                st.success(f"Training completed! Best model identified: **{best_name}** 🏆")

        # Display Results Table if available
        results = st.session_state['training_results']
        if results is not None:
            st.markdown("---")
            st.subheader("📊 Training Results Summary")
            comp_df = build_comparison_dataframe(results, prob_type)
            st.dataframe(comp_df, use_container_width=True)

            best_name = st.session_state['best_model_name']
            st.info(f"🏆 **Winner:** **{best_name}** achieved the top score across the validation holdout split.")

            # Save Model Section
            st.markdown("---")
            st.subheader("💾 Save Trained Model Pipeline")
            save_col1, save_col2 = st.columns([2, 1])
            with save_col1:
                model_to_save = st.selectbox("Select Model to Save", options=list(results.keys()), index=list(results.keys()).index(best_name) if best_name in results else 0)
            with save_col2:
                st.markdown("<br/>", unsafe_allow_html=True)
                if st.button("Save to Disk (Joblib)"):
                    save_dir = os.path.join(CURRENT_DIR, 'saved_models')
                    filename = f"{model_to_save.replace(' ', '_').lower()}_{int(time.time())}.joblib"
                    save_path = os.path.join(save_dir, filename)

                    save_trained_pipeline(
                        filepath=save_path,
                        model=results[model_to_save]['model'],
                        model_name=model_to_save,
                        preprocessor=prep['preprocessor'],
                        target_encoder=prep['target_encoder'],
                        feature_cols=prep['feature_cols'],
                        numerical_cols=prep['numerical_cols'],
                        categorical_cols=prep['categorical_cols'],
                        target_col=prep['target_col'],
                        problem_type=prob_type,
                        metrics=results[model_to_save]['metrics'],
                        feature_names_out=prep['feature_names_out'],
                        target_classes=prep['target_classes']
                    )
                    st.session_state['saved_model_path'] = save_path
                    st.success(f"Model saved to `{filename}` in `saved_models/`!")


# ==============================================================================
# PAGE 5: MODEL EVALUATION
# ==============================================================================

elif "Model Evaluation" in selected_page:
    st.title("📈 Model Evaluation & Diagnostics")
    st.markdown("Inspect confusion matrices, ROC curves, residuals, actual vs predicted regressions, and feature importance charts.")

    results = st.session_state['training_results']
    prep = st.session_state['preprocessed_data']
    prob_type = st.session_state['problem_type']

    if results is None or prep is None:
        st.warning("Please train models first in **🚀 Model Training**.")
    else:
        best_name = st.session_state['best_model_name']
        model_names = list(results.keys())
        default_idx = model_names.index(best_name) if best_name in model_names else 0

        selected_model = st.selectbox("Select Model for In-Depth Evaluation", model_names, index=default_idx)
        model_data = results[selected_model]
        metrics = model_data.get('metrics', {})

        # Display Metrics Cards
        st.markdown("### 🎯 Performance Metrics")
        if prob_type == 'Classification':
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Accuracy", f"{metrics.get('Accuracy', 0.0)*100:.2f}%")
            c2.metric("Precision", f"{metrics.get('Precision', 0.0):.4f}")
            c3.metric("Recall", f"{metrics.get('Recall', 0.0):.4f}")
            c4.metric("F1 Score", f"{metrics.get('F1 Score', 0.0):.4f}")
            auc_disp = f"{metrics.get('AUC Score', 0.0):.4f}" if isinstance(metrics.get('AUC Score'), (int, float)) else str(metrics.get('AUC Score'))
            c5.metric("AUC Score", auc_disp)
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("R² Score", f"{metrics.get('R2 Score', 0.0):.4f}")
            c2.metric("RMSE", f"{metrics.get('RMSE', 0.0):,.4f}")
            c3.metric("MAE", f"{metrics.get('MAE', 0.0):,.4f}")
            c4.metric("MSE", f"{metrics.get('MSE', 0.0):,.4f}")

        st.markdown("---")
        st.subheader("📊 Interactive Visualizations")

        if prob_type == 'Classification':
            col_v1, col_v2 = st.columns(2)

            with col_v1:
                st.markdown("#### Confusion Matrix")
                if 'confusion_matrix' in metrics:
                    fig_cm = plot_confusion_matrix_plotly(
                        metrics['confusion_matrix'],
                        metrics.get('class_labels', [])
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)

            with col_v2:
                st.markdown("#### ROC Curve")
                roc_fig = plot_roc_curve_plotly(metrics.get('roc_data'), selected_model)
                if roc_fig:
                    st.plotly_chart(roc_fig, use_container_width=True)
                else:
                    st.info("ROC Curve requires class probability estimation (supported for binary classification models).")

            col_v3, col_v4 = st.columns(2)
            with col_v3:
                st.markdown("#### Class Distribution (Actual vs Predicted)")
                fig_dist = plot_prediction_distribution_plotly(
                    prep['y_test'],
                    model_data['y_pred'],
                    'Classification',
                    prep['target_classes']
                )
                st.plotly_chart(fig_dist, use_container_width=True)

            with col_v4:
                st.markdown("#### Feature Importance")
                if model_data.get('feature_importance') is not None:
                    fig_feat = plot_feature_importance_plotly(
                        prep['feature_names_out'],
                        model_data['feature_importance'],
                        top_n=10,
                        title=f"{selected_model} - Top Features"
                    )
                    st.plotly_chart(fig_feat, use_container_width=True)
                else:
                    st.info(f"{selected_model} does not natively produce linear coefficients or tree feature importances.")

        else:
            # Regression Visualizations
            col_r1, col_r2 = st.columns(2)

            with col_r1:
                st.markdown("#### Actual vs Predicted Values")
                fig_reg = plot_actual_vs_predicted_plotly(
                    prep['y_test'],
                    model_data['y_pred'],
                    selected_model
                )
                st.plotly_chart(fig_reg, use_container_width=True)

            with col_r2:
                st.markdown("#### Residuals (Error Distribution)")
                fig_err = plot_prediction_distribution_plotly(
                    prep['y_test'],
                    model_data['y_pred'],
                    'Regression'
                )
                st.plotly_chart(fig_err, use_container_width=True)

            col_r3, col_r4 = st.columns(2)
            with col_r3:
                st.markdown("#### Feature Importance / Coefficients")
                if model_data.get('feature_importance') is not None:
                    fig_feat = plot_feature_importance_plotly(
                        prep['feature_names_out'],
                        model_data['feature_importance'],
                        top_n=10,
                        title=f"{selected_model} - Top Predictors"
                    )
                    st.plotly_chart(fig_feat, use_container_width=True)
                else:
                    st.info(f"{selected_model} does not expose feature importances.")

            with col_r4:
                st.markdown("#### Dataset Correlation Heatmap")
                fig_corr = plot_correlation_heatmap_plotly(st.session_state['df_raw'])
                if fig_corr:
                    st.plotly_chart(fig_corr, use_container_width=True)


# ==============================================================================
# PAGE 6: MODEL COMPARISON
# ==============================================================================

elif "Model Comparison" in selected_page:
    st.title("⚖️ Model Comparison & Benchmark")
    st.markdown("Compare all trained machine learning models, identify the winning architecture, and explore algorithmic performance insights.")

    results = st.session_state['training_results']
    prep = st.session_state['preprocessed_data']
    prob_type = st.session_state['problem_type']
    best_name = st.session_state['best_model_name']

    if results is None:
        st.warning("Please train models first in **🚀 Model Training**.")
    else:
        comp_df = build_comparison_dataframe(results, prob_type)

        st.subheader("🏆 Model Benchmark Leaderboard")
        st.dataframe(comp_df, use_container_width=True)

        st.markdown("---")
        # Multi-metric comparative bar chart
        st.subheader("📊 Cross-Algorithm Metric Comparison")
        fig_bar = plot_model_comparison_bar(comp_df, prob_type)
        st.plotly_chart(fig_bar, use_container_width=True)

        # Radar chart for classification
        if prob_type == 'Classification':
            fig_radar = plot_radar_comparison(comp_df, prob_type)
            if fig_radar:
                st.plotly_chart(fig_radar, use_container_width=True)

        # Algorithmic Narrative
        st.markdown("---")
        st.subheader("💡 Algorithmic Performance Explanation")
        explanation = explain_best_model(best_name, comp_df, prob_type)
        st.markdown(explanation)


# ==============================================================================
# PAGE 7: MAKE PREDICTIONS
# ==============================================================================

elif "Make Predictions" in selected_page:
    st.title("🔮 Make Predictions")
    st.markdown("Predict outcomes for individual records via interactive input fields or upload a batch dataset for bulk scoring.")

    results = st.session_state['training_results']
    prep = st.session_state['preprocessed_data']
    best_name = st.session_state['best_model_name']
    prob_type = st.session_state['problem_type']

    if results is None or prep is None or best_name is None:
        st.warning("No trained models available. Please train your models in **🚀 Model Training** first.")
    else:
        # Allow selecting which model to predict with (defaults to best model)
        model_names = list(results.keys())
        predict_model_name = st.selectbox(
            "Select Model for Inference",
            model_names,
            index=model_names.index(best_name) if best_name in model_names else 0
        )
        active_model = results[predict_model_name]['model']

        tab_single, tab_batch = st.tabs(["Single Record Manual Prediction", "Bulk Batch Prediction (CSV/Excel)"])

        # TAB 1: Single Prediction
        with tab_single:
            st.subheader("Enter Feature Values")
            raw_features = prep['feature_cols']
            df_sample = prep['X_train_raw']

            # Build responsive grid for input fields
            num_cols_per_row = 3
            input_values = {}

            # Partition features into rows
            feature_chunks = [raw_features[i:i + num_cols_per_row] for i in range(0, len(raw_features), num_cols_per_row)]

            for chunk in feature_chunks:
                cols = st.columns(len(chunk))
                for idx, feat in enumerate(chunk):
                    with cols[idx]:
                        is_num = feat in prep['numerical_cols']
                        if is_num:
                            series = df_sample[feat].dropna()
                            f_min = float(series.min()) if not series.empty else 0.0
                            f_max = float(series.max()) if not series.empty else 1000.0
                            f_mean = float(series.median()) if not series.empty else 0.0
                            is_int = pd.api.types.is_integer_dtype(series)

                            if is_int:
                                val = st.number_input(
                                    f"{feat}",
                                    value=int(f_mean),
                                    step=1
                                )
                            else:
                                val = st.number_input(
                                    f"{feat}",
                                    value=round(f_mean, 2),
                                    step=round((f_max - f_min) / 100, 2) if f_max > f_min else 0.1
                                )
                            input_values[feat] = val
                        else:
                            # Categorical selectbox
                            unique_opts = sorted(list(df_sample[feat].dropna().unique()))
                            if not unique_opts:
                                unique_opts = ["Missing"]
                            val = st.selectbox(f"{feat}", options=unique_opts)
                            input_values[feat] = val

            if st.button("⚡ Generate Prediction", use_container_width=True):
                try:
                    pred_res = predict_single_record(
                        input_dict=input_values,
                        model=active_model,
                        preprocessor=prep['preprocessor'],
                        target_encoder=prep['target_encoder'],
                        problem_type=prob_type,
                        feature_cols=prep['feature_cols']
                    )

                    pred_out = pred_res['prediction']
                    st.markdown(f"""
                        <div class="prediction-box">
                            <div class="kpi-title">Predicted Outcome ({prep['target_col']})</div>
                            <h2>{pred_out}</h2>
                            <p style="color: #64748B; margin: 0;">Predicted by <b>{predict_model_name}</b></p>
                        </div>
                    """, unsafe_allow_html=True)

                    # Probabilities if classification
                    if prob_type == 'Classification' and pred_res.get('probabilities'):
                        st.markdown("#### Confidence Breakdown")
                        prob_df = pd.DataFrame(
                            list(pred_res['probabilities'].items()),
                            columns=['Class', 'Probability (%)']
                        )
                        st.dataframe(prob_df, use_container_width=True)

                    # Log to recent predictions
                    rec_entry = {**input_values, f"Predicted_{prep['target_col']}": pred_out, "Model": predict_model_name}
                    st.session_state['recent_predictions'].append(rec_entry)

                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")

        # TAB 2: Batch Prediction
        with tab_batch:
            st.subheader("Bulk Dataset Scoring")
            batch_file = st.file_uploader("Upload Test Dataset (CSV / Excel)", type=['csv', 'xlsx', 'xls'], key="batch_upload")

            # Option to use sample test set
            c_b1, c_b2 = st.columns([1, 1])
            with c_b1:
                use_sample_test = st.checkbox("Or test using sample test records", value=False)

            df_batch = None
            if batch_file is not None:
                df_batch = load_dataset(batch_file)
            elif use_sample_test:
                sample_test_path = os.path.join(CURRENT_DIR, 'dataset', 'sample_churn_batch_test.csv' if prob_type == 'Classification' else 'sample_house_batch_test.csv')
                if os.path.exists(sample_test_path):
                    df_batch = pd.read_csv(sample_test_path)

            if df_batch is not None:
                st.markdown(f"Loaded **{len(df_batch)}** batch records for prediction.")
                st.dataframe(df_batch.head(5), use_container_width=True)

                if st.button("⚡ Score Batch Records", use_container_width=True):
                    with st.spinner("Scoring dataset..."):
                        scored_df, err = predict_batch(
                            df_batch=df_batch,
                            model=active_model,
                            preprocessor=prep['preprocessor'],
                            target_encoder=prep['target_encoder'],
                            problem_type=prob_type,
                            feature_cols=prep['feature_cols'],
                            target_col=prep['target_col']
                        )

                        if err:
                            st.error(err)
                        else:
                            st.success(f"Batch prediction completed for all {len(scored_df)} rows!")
                            st.dataframe(scored_df, use_container_width=True)

                            # Download CSV
                            csv_data = scored_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Download Prediction Results (CSV)",
                                data=csv_data,
                                file_name=f"batch_predictions_{prep['target_col']}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )


# ==============================================================================
# PAGE 8: DOWNLOAD REPORTS
# ==============================================================================

elif "Download Reports" in selected_page:
    st.title("📥 Export Deliverables & Reports")
    st.markdown("Download serialized models, evaluation metrics, comparison summaries, prediction batches, and comprehensive PDF reports.")

    results = st.session_state['training_results']
    prep = st.session_state['preprocessed_data']
    df_raw = st.session_state['df_raw']
    best_name = st.session_state['best_model_name']
    prob_type = st.session_state['problem_type']

    if results is None or prep is None or best_name is None:
        st.warning("Please complete model training first before generating deliverables.")
    else:
        comp_df = build_comparison_dataframe(results, prob_type)
        overview = get_dataset_overview(df_raw) if df_raw is not None else {}
        explanation = explain_best_model(best_name, comp_df, prob_type)

        d_col1, d_col2 = st.columns(2)

        # 1. PDF Project Report
        with d_col1:
            st.markdown("""
                <div class="kpi-card" style="margin-bottom: 20px;">
                    <div class="kpi-title">Executive Audit</div>
                    <h3 style="color: #1E3A8A; margin: 4px 0 10px 0;">📄 Complete PDF Project Report</h3>
                    <p style="color: #64748B; font-size: 0.85rem;">Professional executive report with dataset summaries, preprocessing audit, algorithmic leaderboard, and architectural rationale.</p>
                </div>
            """, unsafe_allow_html=True)

            try:
                pdf_bytes = generate_pdf_report(
                    dataset_name=st.session_state['dataset_name'],
                    target_col=prep['target_col'],
                    problem_type=prob_type,
                    df_comp=comp_df,
                    best_model_name=best_name,
                    dataset_summary=overview,
                    preprocessing_summary=prep,
                    model_explanation=explanation
                )

                st.download_button(
                    label="📥 Download PDF Project Report",
                    data=pdf_bytes,
                    file_name=f"ML_Project_Report_{prep['target_col']}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Failed to compile PDF: {str(e)}")

        # 2. Trained Model Bundle (.joblib)
        with d_col2:
            st.markdown("""
                <div class="kpi-card" style="margin-bottom: 20px;">
                    <div class="kpi-title">Model Serialization</div>
                    <h3 style="color: #1E3A8A; margin: 4px 0 10px 0;">📦 Trained Model Bundle (Joblib)</h3>
                    <p style="color: #64748B; font-size: 0.85rem;">Self-contained bundle including fitted model, preprocessor pipeline, encoders, and feature schema ready for production inference.</p>
                </div>
            """, unsafe_allow_html=True)

            buf = io.BytesIO()
            import joblib
            bundle = {
                'model': results[best_name]['model'],
                'model_name': best_name,
                'preprocessor': prep['preprocessor'],
                'target_encoder': prep['target_encoder'],
                'feature_cols': prep['feature_cols'],
                'target_col': prep['target_col'],
                'problem_type': prob_type,
                'metrics': results[best_name]['metrics']
            }
            joblib.dump(bundle, buf)
            joblib_bytes = buf.getvalue()

            st.download_button(
                label=f"📥 Download {best_name} (.joblib)",
                data=joblib_bytes,
                file_name=f"{best_name.replace(' ', '_').lower()}_pipeline.joblib",
                mime="application/octet-stream",
                use_container_width=True
            )

        st.markdown("---")
        d_col3, d_col4 = st.columns(2)

        # 3. Model Comparison CSV
        with d_col3:
            st.markdown("""
                <div class="kpi-card" style="margin-bottom: 20px;">
                    <div class="kpi-title">Benchmark Summary</div>
                    <h3 style="color: #1E3A8A; margin: 4px 0 10px 0;">📊 Model Comparison Table (CSV)</h3>
                    <p style="color: #64748B; font-size: 0.85rem;">CSV file containing ranked algorithms, training times, and all evaluation metrics.</p>
                </div>
            """, unsafe_allow_html=True)

            comp_csv = comp_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Comparison CSV",
                data=comp_csv,
                file_name=f"model_comparison_{prep['target_col']}.csv",
                mime="text/csv",
                use_container_width=True
            )

        # 4. Evaluation Metrics JSON
        with d_col4:
            st.markdown("""
                <div class="kpi-card" style="margin-bottom: 20px;">
                    <div class="kpi-title">Raw Diagnostics</div>
                    <h3 style="color: #1E3A8A; margin: 4px 0 10px 0;">🔬 Evaluation Metrics (JSON)</h3>
                    <p style="color: #64748B; font-size: 0.85rem;">Detailed JSON dump containing confusion matrices, ROC curve thresholds, and performance scores.</p>
                </div>
            """, unsafe_allow_html=True)

            # Extract clean serializable metrics
            metrics_export = {}
            for m_name, m_data in results.items():
                if 'metrics' in m_data:
                    metrics_export[m_name] = m_data['metrics']

            json_bytes = json.dumps(metrics_export, indent=2).encode('utf-8')
            st.download_button(
                label="📥 Download Evaluation Metrics (JSON)",
                data=json_bytes,
                file_name=f"evaluation_metrics_{prep['target_col']}.json",
                mime="application/json",
                use_container_width=True
            )
