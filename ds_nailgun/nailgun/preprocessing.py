"""
Preprocessing Pipeline Module

Creates scikit-learn preprocessing pipelines based on YAML configuration.
"""

import yaml
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    LabelEncoder,
    MinMaxScaler,
    RobustScaler,
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_selection import (
    SelectKBest,
    VarianceThreshold,
    RFE,
    f_classif,
    chi2,
    mutual_info_classif,
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


class PreprocessingPipeline:
    """Creates and manages preprocessing pipelines based on config."""

    def __init__(self, config_path):
        """Initialize with YAML config file path."""
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        self.pipeline = None
        self.feature_columns = self._get_feature_columns()

    def _get_feature_columns(self):
        """Get feature columns organized by type."""
        features = self.config["data"]["features"]
        return {
            "int": features.get("int", []),
            "float": features.get("float", []),
            "categorical": features.get("categorical", []),
            "string": features.get("string", []),
        }

    def create_pipeline(self):
        """Create the preprocessing pipeline based on config."""
        preprocessing_config = self.config.get("preprocessing", {})

        # Create transformers list for ColumnTransformer
        transformers = []

        # Handle different feature types with combined imputation + transformation
        if "imputation" in preprocessing_config:
            imputation_config = preprocessing_config["imputation"]

            # Handle integer features
            if "int" in imputation_config and self.feature_columns["int"]:
                cols = self.feature_columns["int"]
                method = imputation_config["int"]["method"]

                if method == "constant":
                    fill_value = imputation_config["int"].get("fill_value", 0)
                    imputer = SimpleImputer(strategy="constant", fill_value=fill_value)
                else:
                    imputer = SimpleImputer(strategy=method)

                # Int features typically don't need scaling, just imputation
                transformers.append(("int_pipeline", imputer, cols))

            # Handle float features
            if "float" in imputation_config and self.feature_columns["float"]:
                cols = self.feature_columns["float"]
                impute_method = imputation_config["float"]["method"]

                if impute_method == "constant":
                    fill_value = imputation_config["float"].get("fill_value", 0.0)
                    imputer = SimpleImputer(strategy="constant", fill_value=fill_value)
                else:
                    imputer = SimpleImputer(strategy=impute_method)

                # Check if we need scaling
                transform_config = preprocessing_config.get("transforms", {})
                if "float" in transform_config:
                    method = transform_config["float"]["method"]

                    if method == "standard_scaler":
                        scaler = StandardScaler()
                    elif method == "min_max_scaler":
                        scaler = MinMaxScaler()
                    elif method == "robust_scaler":
                        scaler = RobustScaler()
                    else:
                        scaler = FunctionTransformer(lambda x: x)  # No-op

                    # Create pipeline: impute -> scale
                    float_pipeline = Pipeline(
                        [("imputer", imputer), ("scaler", scaler)]
                    )
                    transformers.append(("float_pipeline", float_pipeline, cols))
                else:
                    # Just imputation
                    transformers.append(("float_imputer", imputer, cols))

            # Handle categorical features
            if (
                "categorical" in imputation_config
                and self.feature_columns["categorical"]
            ):
                cols = self.feature_columns["categorical"]
                impute_method = imputation_config["categorical"]["method"]

                if impute_method == "constant":
                    fill_value = imputation_config["categorical"].get(
                        "fill_value", "missing"
                    )
                    # Ensure fill_value is a string for categorical data
                    if not isinstance(fill_value, str):
                        fill_value = str(fill_value)
                    imputer = SimpleImputer(strategy="constant", fill_value=fill_value)
                else:
                    imputer = SimpleImputer(strategy=impute_method)

                # Check if we need encoding
                transform_config = preprocessing_config.get("transforms", {})
                if "categorical" in transform_config:
                    method = transform_config["categorical"]["method"]

                    if method == "one_hot_encoding":
                        encoder = OneHotEncoder(sparse_output=False, drop="first")
                        # Create pipeline: impute -> encode
                        cat_pipeline = Pipeline(
                            [("imputer", imputer), ("encoder", encoder)]
                        )
                        transformers.append(
                            ("categorical_pipeline", cat_pipeline, cols)
                        )
                    elif method == "label_encoding":
                        # Label encoding needs special handling
                        encoder = LabelEncoder()
                        # For now, just impute
                        transformers.append(("categorical_imputer", imputer, cols))
                    else:
                        # Just imputation
                        transformers.append(("categorical_imputer", imputer, cols))
                else:
                    # Just imputation
                    transformers.append(("categorical_imputer", imputer, cols))

        # Create initial column transformer for preprocessing
        if transformers:
            column_transformer = ColumnTransformer(
                transformers=transformers,
                remainder="drop",  # Drop columns not specified in transformers
            )
        else:
            # No transformations specified
            column_transformer = FunctionTransformer(lambda x: x)

        # Handle feature selection if specified
        if "feature_selection" in preprocessing_config:
            feature_select_config = preprocessing_config["feature_selection"]
            method = feature_select_config.get("method")

            if method:
                # Build a pipeline with preprocessing followed by feature selection
                pipeline_steps = [("preprocessing", column_transformer)]

                if method == "select_k_best":
                    # Get parameters for SelectKBest
                    params = feature_select_config.get("params", {})
                    k = params.get("k", 5)
                    score_func_name = params.get("score_func", "f_classif")

                    # Map score function names to actual functions
                    score_funcs = {
                        "f_classif": f_classif,
                        "chi2": chi2,
                        "mutual_info": mutual_info_classif,
                    }
                    score_func = score_funcs.get(score_func_name, f_classif)

                    # Create SelectKBest selector
                    selector = SelectKBest(score_func=score_func, k=k)
                    pipeline_steps.append(("feature_selection", selector))

                elif method == "variance_threshold":
                    # Get parameters for VarianceThreshold
                    params = feature_select_config.get("params", {})
                    threshold = params.get("threshold", 0.0)

                    # Create VarianceThreshold selector
                    selector = VarianceThreshold(threshold=threshold)
                    pipeline_steps.append(("feature_selection", selector))

                elif method == "rfe":
                    # Get parameters for RFE
                    params = feature_select_config.get("params", {})
                    n_features = params.get("n_features", 5)
                    step = params.get("step", 1)
                    estimator_name = params.get("estimator", "logistic_regression")

                    # Create estimator for RFE
                    if estimator_name == "logistic_regression":
                        estimator = LogisticRegression(max_iter=1000)
                    else:
                        # Default to logistic regression
                        estimator = LogisticRegression(max_iter=1000)

                    # Create RFE selector
                    selector = RFE(
                        estimator=estimator, n_features_to_select=n_features, step=step
                    )
                    pipeline_steps.append(("feature_selection", selector))

                # Create the full pipeline
                self.pipeline = Pipeline(steps=pipeline_steps)
            else:
                # No feature selection method specified, just use column transformer
                self.pipeline = column_transformer
        else:
            # No feature selection specified, just use column transformer
            self.pipeline = column_transformer

        return self.pipeline


def create_preprocessing_pipeline(config_path):
    """
    Convenience function to create a preprocessing pipeline.

    Args:
        config_path (str): Path to YAML config file

    Returns:
        PreprocessingPipeline: Configured preprocessing pipeline
    """
    return PreprocessingPipeline(config_path)


# Test the preprocessing pipeline
if __name__ == "__main__":
    import time

    # Test with titanic config
    config_path = "ds_nailgun/configs/examples/titanic_data_config.yaml"

    # Create pipeline
    start_time = time.time()
    preprocessor = create_preprocessing_pipeline(config_path)
    pipeline = preprocessor.create_pipeline()
    setup_time = time.time() - start_time

    print(f"Pipeline created in {setup_time:.4f} seconds")
    print(f"Pipeline type: {type(pipeline).__name__}")

    if isinstance(pipeline, Pipeline):
        print(f"Pipeline steps: {len(pipeline.steps)}")
        for name, step in pipeline.steps:
            print(f"  - {name}: {type(step).__name__}")
    elif isinstance(pipeline, ColumnTransformer):
        print(f"Pipeline steps: {len(pipeline.transformers)}")
        for name, transformer, columns in pipeline.transformers:
            print(f"  - {name}: {type(transformer).__name__} on {columns}")
    else:
        print("Pipeline: No-op transformer")

    print("Preprocessing pipeline ready for use!")
    print("To use: pipeline.fit_transform(X_train)")
