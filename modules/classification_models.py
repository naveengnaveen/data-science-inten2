"""
Classification Models Module for Predictive Modeling Dashboard.
Defines the 5 required supervised classification algorithms with standard
scikit-learn implementations and rich metadata.
"""

from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


def get_classification_models(random_state: int = 42) -> Dict[str, Dict[str, Any]]:
    """
    Returns a dictionary of all 5 supervised classification algorithms
    ready for training, along with documentation and properties.
    """
    return {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=random_state),
            "description": "Linear model for binary/multiclass classification using the logistic sigmoid function.",
            "type": "Linear / Probabilistic",
            "supports_feature_importance": True,
            "importance_attr": "coef_"
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(random_state=random_state, max_depth=10),
            "description": "Non-parametric tree-based model that splits features based on Gini impurity or entropy.",
            "type": "Tree-based",
            "supports_feature_importance": True,
            "importance_attr": "feature_importances_"
        },
        "Random Forest": {
            "model": RandomForestClassifier(n_estimators=100, random_state=random_state),
            "description": "Ensemble of diverse decision trees using bagging and feature subsampling for high variance reduction.",
            "type": "Ensemble",
            "supports_feature_importance": True,
            "importance_attr": "feature_importances_"
        },
        "Support Vector Machine (SVM)": {
            "model": SVC(probability=True, random_state=random_state, kernel='rbf'),
            "description": "Finds the optimal separating hyperplane in high-dimensional kernel space with probability calibration.",
            "type": "Kernel / Margin-based",
            "supports_feature_importance": False,
            "importance_attr": None
        },
        "K-Nearest Neighbors (KNN)": {
            "model": KNeighborsClassifier(n_neighbors=5),
            "description": "Instance-based lazy learner assigning class based on majority vote of the k closest Euclidean neighbors.",
            "type": "Instance-based",
            "supports_feature_importance": False,
            "importance_attr": None
        }
    }
