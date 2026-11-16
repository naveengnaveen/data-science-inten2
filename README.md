# ⚡ Predictive Modeling Using Machine Learning

An end-to-end, production-grade supervised machine learning application and interactive Streamlit dashboard. This platform automates the complete data science lifecycle—from dataset ingestion and exploratory profiling to leak-free preprocessing, multi-algorithm training, evaluation diagnostics, real-time manual & batch inference, and automated PDF executive reporting.

---

## 🌟 Key Features

### 1. 📁 Dataset Ingestion & Dynamic Target Selection
- Supports **CSV** and **Excel** (`.xlsx`, `.xls`) file formats.
- High-level overview: Total rows, columns, memory usage, duplicate records, and missing percentage.
- Column-wise schema table detailing data types, null counts, and cardinality.
- Interactive previews for head (first 5 records) and tail (last 5 records).
- **Dynamic Target Selection**: Select any column as the prediction target.
- **Automated Task Inference**: Intelligently determines whether the problem is **Classification** or **Regression** with heuristic explanations, while allowing full user confirmation/override.
- **Identifier Detection**: Automatically flags high-cardinality ID/key columns (e.g., `CustomerID`) and excludes them from model training to prevent spurious memorization.

### 2. ⚙️ Leak-Free Data Preprocessing Pipelines
- **Data Leakage Prevention**: Splits data into train and test sets *before* fitting any transformer or computing summary statistics.
- **Missing Value Imputation**: Median, Mean, or Most Frequent for numerical features; Mode or Constant for categorical features.
- **Deduplication**: Automatic detection and removal of duplicate rows.
- **Categorical Encoding**: One-Hot Encoding (with `handle_unknown='ignore'`) or Ordinal Encoding.
- **Numerical Scaling**: `StandardScaler`, `MinMaxScaler`, `RobustScaler`, or Unscaled.
- **Outlier Detection & Treatment**: IQR-based boundary detection with leak-free clipping to $[Q_1 - 1.5 \times IQR, Q_3 + 1.5 \times IQR]$ fitted strictly on training data.
- Configurable **Train-Test Split Ratio** (10% to 40%, default 80:20).

### 3. 🤖 Supervised Machine Learning Algorithms

| Task Type | Implemented Algorithms |
| :--- | :--- |
| **Classification** | 1. **Logistic Regression**<br/>2. **Decision Tree Classifier**<br/>3. **Random Forest Classifier**<br/>4. **Support Vector Machine (SVM)** (RBF Kernel & Calibrated Probabilities)<br/>5. **K-Nearest Neighbors (KNN)** |
| **Regression** | 1. **Linear Regression** (Ordinary Least Squares)<br/>2. **Decision Tree Regressor**<br/>3. **Random Forest Regressor**<br/>4. **Support Vector Regressor (SVR)**<br/>5. **K-Nearest Neighbors Regressor** |

### 4. 🚀 Automated Training & Best Model Identification
- Concurrent multi-model training with progress bar indicators.
- Latency tracking: Training execution time and prediction throughput.
- **Automated Winner Selection**:
  - **Classification**: Selects model with highest **F1 Score** (tiebreaker: Accuracy).
  - **Regression**: Selects model with highest **$R^2$ Score** (tiebreaker: lowest RMSE).

### 5. 📈 In-Depth Evaluation & Interactive Diagnostics
- **Classification Metrics**: Accuracy, Precision, Recall, F1 Score, Confusion Matrix, ROC Curve, and AUC Score.
- **Regression Metrics**: Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and $R^2$ Score.
- **Interactive Visualizations (Plotly)**:
  - Confusion Matrix annotated heatmap.
  - ROC Curve with AUC score and random chance benchmark.
  - Actual vs. Predicted scatter plot with perfect fit $y = x$ line.
  - Residuals and error distribution histogram.
  - Feature Importance / Coefficient bar charts.
  - Feature Correlation heatmap.
  - Class distribution comparison (Actual vs. Predicted).

### 6. ⚖️ Algorithmic Leaderboard & Explanation Engine
- Comprehensive benchmark comparison table.
- Multi-metric comparative grouped bar charts.
- Multivariate radar / spider chart for classification models.
- **Automated Algorithmic Explanations**: Generates natural language insights detailing why the winning algorithm outperformed the others based on variance reduction, decision boundaries, and loss functions.

### 7. 🔮 Real-Time Predictions Module
- **Single Record Manual Prediction**:
  - Responsive dynamic input form matching feature datatypes, ranges, and categories.
  - Confidence probability breakdown for classification.
  - Recent prediction activity log.
- **Bulk Batch Prediction**:
  - Upload CSV/Excel batch dataset for bulk scoring.
  - Instant schema validation.
  - Appends predictions and confidence scores.
  - Downloadable prediction results CSV.

### 8. 📥 Comprehensive Report & Deliverable Downloads
- **Executive PDF Project Report**: Formatted multi-page PDF generated via ReportLab containing project summary, preprocessing audit, comparison table, and winning model analysis.
- **Trained Model Bundle (`.joblib`)**: Self-contained bundle with fitted model, preprocessing pipeline, encoders, and feature schema.
- **Model Comparison Table (`.csv`)**.
- **Diagnostic Metrics (`.json`)**.
- **Scored Predictions (`.csv`)**.

---

## 🛠️ Technology Stack

- **Language:** Python 3.10+
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Data Visualization:** Plotly, Matplotlib, Seaborn
- **Dashboard UI:** Streamlit
- **Model Serialization:** Joblib
- **Document Generation:** ReportLab
- **Development Environment:** VS Code

---

## 📁 Project Structure

```text
predictive-modeling-ml/
│
├── app.py                      # Main Streamlit Dashboard Application
├── requirements.txt            # Python Dependencies
├── README.md                   # Project Documentation
│
├── modules/                    # Modular Machine Learning Architecture
│   ├── data_loader.py          # Data ingestion & schema profiling
│   ├── preprocessing.py        # Leak-free ColumnTransformers & IQR clippers
│   ├── classification_models.py# 5 Classification algorithms & metadata
│   ├── regression_models.py    # 5 Regression algorithms & metadata
│   ├── model_training.py       # Training loops, selection & Joblib saving
│   ├── model_evaluation.py     # Evaluation metrics & Plotly visualizers
│   ├── model_comparison.py     # Benchmark tables & explanation generator
│   ├── prediction.py           # Single & bulk batch prediction logic
│   └── report_generator.py     # ReportLab PDF & export generator
│
├── dataset/                    # Sample Datasets
│   ├── sample_dataset.csv      # Customer Churn (Classification)
│   ├── sample_housing_dataset.csv # House Price (Regression)
│   ├── sample_churn_batch_test.csv# Test batch for classification
│   └── sample_house_batch_test.csv# Test batch for regression
│
├── saved_models/               # Serialized Joblib model pipelines
├── outputs/                    # Output artifacts
└── assets/                     # UI styling assets & documentation
```

---

## 🚀 Installation & Execution

### 1. Clone or Open the Workspace in VS Code
Open the project directory in VS Code or your terminal:
```bash
cd "predictive-modeling-ml"
```

### 2. Install Required Dependencies
Ensure you have Python 3.10+ installed, then run:
```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit Dashboard
Run the Streamlit application:
```bash
streamlit run app.py
```

The application will launch in your default web browser at `http://localhost:8501`.

---

## 📖 Step-by-Step Workflow Guide

1. **Dashboard Overview**: View global workspace stats and recent activities.
2. **Upload Dataset**: Upload your CSV/Excel file or click **"👥 Churn"** or **"🏡 Housing"** in the sidebar to load the included sample datasets. Select your target column.
3. **Data Preprocessing**: Configure train/test ratio, imputation strategies, scaling, and outlier handling. Click **"🚀 Run Preprocessing Pipeline"**.
4. **Model Training**: Select algorithms to benchmark and click **"🏋️ Train Selected Models"**. Save your winning model bundle to disk.
5. **Model Evaluation**: Inspect confusion matrices, ROC curves, residuals, and feature importance charts for each model.
6. **Model Comparison**: Review the algorithmic leaderboard, multi-metric bar charts, radar charts, and natural language explanations.
7. **Make Predictions**: Test single record manual inputs or upload a test dataset for bulk batch scoring.
8. **Download Reports**: Export your trained `.joblib` model pipeline, prediction CSVs, and complete executive PDF project report!

---

## 📄 License
This project is open-source and intended for academic, research, and production data science workflows.
