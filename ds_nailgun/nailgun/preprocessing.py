"""
Preprocessing Pipeline Module

Creates scikit-learn preprocessing pipelines based on YAML configuration.
"""

import yaml
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
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
        """Get feature columns organized by user-defined feature groups."""
        features = self.config["data"]["features"]
        # Return all feature groups as defined in config, no hardcoded types
        return features

    def _create_feature_pipeline(
        self, feature_group, imputation_config, transform_config
    ):
        """Create a preprocessing pipeline for a specific user-defined feature group."""
        if feature_group not in imputation_config:
            return None

        cols = self.feature_columns.get(feature_group, [])
        if not cols:
            return None

        impute_method = imputation_config[feature_group]["method"]

        # Create imputer based on config
        if impute_method == "constant":
            fill_value = imputation_config[feature_group].get("fill_value", "missing")
            # For categorical-like features, ensure fill_value is appropriate
            if feature_group.lower() in ["categorical", "string"] and not isinstance(
                fill_value, str
            ):
                fill_value = str(fill_value)
            imputer = SimpleImputer(strategy="constant", fill_value=fill_value)
        else:
            imputer = SimpleImputer(strategy=impute_method)

        # Apply transforms if specified in config
        if transform_config and feature_group in transform_config:
            method = transform_config[feature_group]["method"]

            if method == "standard_scaler":
                transformer = Pipeline(
                    [("imputer", imputer), ("scaler", StandardScaler())]
                )
            elif method == "min_max_scaler":
                transformer = Pipeline(
                    [("imputer", imputer), ("scaler", MinMaxScaler())]
                )
            elif method == "robust_scaler":
                transformer = Pipeline(
                    [("imputer", imputer), ("scaler", RobustScaler())]
                )
            elif method == "one_hot_encoding":
                transformer = Pipeline(
                    [
                        ("imputer", imputer),
                        ("encoder", OneHotEncoder(sparse_output=False, drop="first")),
                    ]
                )
            else:
                # Unknown method, just impute
                transformer = imputer
        else:
            # No transforms, just impute
            transformer = imputer

        return (f"{feature_group}_pipeline", transformer, cols)

    def _create_imputation_transformers(self, preprocessing_config):
        """Create transformers list for imputation and transforms."""
        transformers = []

        if "imputation" in preprocessing_config:
            imputation_config = preprocessing_config["imputation"]
            transform_config = preprocessing_config.get("transforms", {})

            # Loop through all user-defined feature groups and create pipelines
            for feature_group in self.feature_columns.keys():
                pipeline_info = self._create_feature_pipeline(
                    feature_group, imputation_config, transform_config
                )
                if pipeline_info:
                    transformers.append(pipeline_info)

        return transformers

    def _create_feature_selection_step(self, preprocessing_config, base_transformer):
        """Create feature selection step if specified."""
        if "feature_selection" not in preprocessing_config:
            return base_transformer

        feature_select_config = preprocessing_config["feature_selection"]
        method = feature_select_config.get("method")

        if not method:
            return base_transformer

        # Build a pipeline with preprocessing followed by feature selection
        pipeline_steps = [("preprocessing", base_transformer)]

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
        return Pipeline(steps=pipeline_steps)

    def create_pipeline(self):
        """
        Create the preprocessing pipeline based on config.

        Returns:
            Pipeline, ColumnTransformer, or FunctionTransformer: The configured preprocessing pipeline.
        """
        preprocessing_config = self.config.get("preprocessing", {})

        # Create imputation and transform transformers
        transformers = self._create_imputation_transformers(preprocessing_config)

        # Create base transformer (ColumnTransformer or passthrough)
        if transformers:
            base_transformer = ColumnTransformer(
                transformers=transformers, remainder="drop"
            )
        else:
            base_transformer = FunctionTransformer(func=None)

        # Add feature selection if specified
        final_pipeline = self._create_feature_selection_step(
            preprocessing_config, base_transformer
        )

        return final_pipeline


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
