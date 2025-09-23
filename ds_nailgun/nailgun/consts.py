"""
Constants for the nailgun module.
"""

from sklearn.feature_selection import (
    f_classif,
    f_regression,
    chi2,
    mutual_info_classif,
    mutual_info_regression,
)

# Map score function names to actual functions for feature selection
SCORE_FUNCS = {
    "f_classif": f_classif,
    "f_regression": f_regression,
    "chi2": chi2,
    "mutual_info": mutual_info_classif,  # backward compatibility
    "mutual_info_classif": mutual_info_classif,
    "mutual_info_regression": mutual_info_regression,
}

# Default values for preprocessing
DEFAULT_K_BEST = 5
DEFAULT_VARIANCE_THRESHOLD = 0.0
DEFAULT_RFE_N_FEATURES = 5
DEFAULT_RFE_STEP = 1
DEFAULT_LOGISTIC_REGRESSION_MAX_ITER = 1000
DEFAULT_CONSTANT_FILL_VALUE = "missing"

# Method names for preprocessing
PREPROCESSING_METHODS = {
    "imputation": ["constant", "mean", "median", "most_frequent"],
    "transforms": [
        "standard_scaler",
        "min_max_scaler",
        "robust_scaler",
        "one_hot_encoding",
    ],
    "feature_selection": ["select_k_best", "variance_threshold", "rfe"],
}

# Estimator names
ESTIMATOR_NAMES = ["logistic_regression"]

# Feature group names that use string fill values
STRING_FEATURE_GROUPS = ["categorical", "string"]

# Controller constants
HYPERTUNING_METHODS = ["grid_search", "random_search"]
SCORING_NAMES = ["pinball_loss"]
SAVE_FORMATS = ["joblib", "pickle"]

# Directory names
DEFAULT_BASE_DIR = "experiments"
MODELS_DIR = "models"
RESULTS_DIR = "results"
LOGS_DIR = "logs"
CONFIGS_DIR = "configs"

# File naming patterns
EXPERIMENT_CONFIG_FILE = "experiment_config.yaml"
EXPERIMENT_SUMMARY_FILE = "experiment_summary.yaml"

# Default values for controller
DEFAULT_CV_FOLDS = 5
DEFAULT_GRID_SEARCH_VERBOSE = 2
DEFAULT_PINBALL_ALPHA = 0.5
DEFAULT_SCORING_NAME = "accuracy"
DEFAULT_MODEL_TYPE = "unknown"

# Logging defaults
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Timestamp format
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
