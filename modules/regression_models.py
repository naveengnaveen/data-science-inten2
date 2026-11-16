"""
Regression Models Module for Predictive Modeling Dashboard.
Defines the 5 required supervised regression algorithms with standard
scikit-learn implementations and rich metadata.
"""

from typing import Dict, Any
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor


def get_regression_models(random_state: int = 42) -> Dict[str, Dict[str, Any]]:
    """
    Returns a dictionary of all 5 supervised regression algorithms
    ready for training, along with documentation and properties.
    """
    return {
        "Linear Regression": {
            "model": LinearRegression(),
            "description": "Ordinary Least Squares regression estimating linear coefficients to minimize residual sum of squares.",
            "type": "Linear",
            "supports_feature_importance": True,
            "importance_attr": "coef_"
        },
        "Decision Tree Regressor": {
            "model": DecisionTreeRegressor(random_state=random_state, max_depth=10),
            "description": "Non-parametric regression tree minimizing mean squared error by partitioning continuous feature space.",
            "type": "Tree-based",
            "supports_feature_importance": True,
            "importance_attr": "feature_importances_"
        },
        "Random Forest Regressor": {
            "model": RandomForestRegressor(n_estimators=100, random_state=random_state),
            "description": "Ensemble of regression trees averaging multiple bootstrap tree predictions to minimize prediction variance.",
            "type": "Ensemble",
            "supports_feature_importance": True,
            "importance_attr": "feature_importances_"
        },
        "Support Vector Regressor (SVR)": {
            "model": SVR(kernel='rbf', C=1.0, epsilon=0.1),
            "description": "Epsilon-insensitive Support Vector Regression mapping inputs into a high-dimensional feature space.",
            "type": "Kernel / Margin-based",
            "supports_feature_importance": False,
            "importance_attr": None
        },
        "K-Nearest Neighbors Regressor": {
            "model": KNeighborsRegressor(n_neighbors=5),
            "description": "Predicts continuous outcome via local interpolation / distance-weighted mean of k nearest neighbors.",
            "type": "Instance-based",
            "supports_feature_importance": False,
            "importance_attr": None
        }
    }
