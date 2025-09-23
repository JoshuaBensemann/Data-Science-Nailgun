"""
Preprocessing Pipeline Module

Creates scikit-learn preprocessing pipelines based on YAML configuration.
"""

import yaml
import os
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
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import logging
from tqdm import tqdm
import joblib
from .consts import (
    SCORE_FUNCS,
    DEFAULT_K_BEST,
    DEFAULT_VARIANCE_THRESHOLD,
    DEFAULT_RFE_N_FEATURES,
    DEFAULT_RFE_STEP,
    DEFAULT_LOGISTIC_REGRESSION_MAX_ITER,
    DEFAULT_CONSTANT_FILL_VALUE,
    STRING_FEATURE_GROUPS,
)


class DataFramePreservingTransformer(BaseEstimator, TransformerMixin):
    """Transformer that preserves pandas DataFrame structure and feature names."""

    def __init__(self, transformer):
        self.transformer = transformer
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names_in_ = X.columns.tolist()
        self.transformer.fit(X, y)
        return self

    def transform(self, X):
        # If input is a DataFrame, preserve structure
        if hasattr(X, "columns") and hasattr(self.transformer, "get_feature_names_out"):
            try:
                # Try to get output feature names
                X_transformed = self.transformer.transform(X)
                if hasattr(X_transformed, "shape") and len(X_transformed.shape) == 2:
                    feature_names = self.transformer.get_feature_names_out(X.columns)
                    return pd.DataFrame(
                        X_transformed, columns=feature_names, index=X.index
                    )
                else:
                    return X_transformed
            except Exception:
                # Fallback to regular transform
                return self.transformer.transform(X)
        else:
            return self.transformer.transform(X)


class PreprocessingPipeline:
    """Creates and manages preprocessing pipelines based on config."""

    def __init__(self, config_path):
        """Initialize with YAML config file path."""
        self.logger = logging.getLogger(__name__)
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)
        self.logger.info(
            f"PreprocessingPipeline initialized with config: {config_path}"
        )

        self.pipeline = None
        self.feature_columns = self._get_feature_columns()
        self.is_regression = (
            self.config.get("data", {}).get("target", {}).get("classes", 1) == 0
        )

    def _get_feature_columns(self):
        """Get feature columns organized by user-defined feature groups."""
        features = self.config["data"]["features"]
        # Return all feature groups as defined in config, no hardcoded types
        self.logger.debug(f"Feature groups found: {list(features.keys())}")
        for group, cols in features.items():
            self.logger.debug(
                f"Feature group '{group}': {len(cols) if isinstance(cols, list) else 'not a list'} columns"
            )
        return features

    def _get_cache_config(self, preprocessing_config):
        """Get cache configuration from preprocessing config."""
        cache_config = preprocessing_config.get("cache", {})

        if not cache_config.get("enabled", False):
            return None

        cache_dir = cache_config.get("directory", ".cache")
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)

        # Create joblib Memory object
        memory = joblib.Memory(location=cache_dir, verbose=0)
        self.logger.info(f"Pipeline caching enabled with directory: {cache_dir}")
        return memory

    def _create_feature_pipeline(
        self, feature_group, imputation_config, transform_config, memory=None
    ):
        """Create a preprocessing pipeline for a specific user-defined feature group."""
        self.logger.debug(f"Creating pipeline for feature group: {feature_group}")
        if feature_group not in imputation_config:
            self.logger.debug(f"No imputation config for {feature_group}, skipping")
            return None

        cols = self.feature_columns.get(feature_group, [])
        if not cols:
            self.logger.debug(f"No columns for feature group {feature_group}, skipping")
            return None

        impute_method = imputation_config[feature_group]["method"]
        self.logger.debug(f"Imputation method for {feature_group}: {impute_method}")

        # Create imputer based on config
        if impute_method == "constant":
            fill_value = imputation_config[feature_group].get(
                "fill_value", DEFAULT_CONSTANT_FILL_VALUE
            )
            # For categorical-like features, ensure fill_value is appropriate
            if feature_group.lower() in STRING_FEATURE_GROUPS and not isinstance(
                fill_value, str
            ):
                fill_value = str(fill_value)
            imputer = SimpleImputer(strategy="constant", fill_value=fill_value)
        else:
            imputer = SimpleImputer(strategy=impute_method)

        # Apply transforms if specified in config
        if transform_config and feature_group in transform_config:
            method = transform_config[feature_group]["method"]
            self.logger.debug(f"Transform method for {feature_group}: {method}")

            if method == "standard_scaler":
                transformer = Pipeline(
                    [("imputer", imputer), ("scaler", StandardScaler())], memory=memory
                )
            elif method == "min_max_scaler":
                transformer = Pipeline(
                    [("imputer", imputer), ("scaler", MinMaxScaler())], memory=memory
                )
            elif method == "robust_scaler":
                transformer = Pipeline(
                    [("imputer", imputer), ("scaler", RobustScaler())], memory=memory
                )
            elif method == "one_hot_encoding":
                transformer = Pipeline(
                    [
                        ("imputer", imputer),
                        ("encoder", OneHotEncoder(sparse_output=False, drop="first")),
                    ],
                    memory=memory,
                )
            else:
                # Unknown method, just impute
                transformer = imputer
        else:
            # No transforms, just impute
            transformer = imputer

        self.logger.debug(
            f"Pipeline created for {feature_group} with {len(cols)} columns"
        )
        return (f"{feature_group}_pipeline", transformer, cols)

    def _create_imputation_transformers(self, preprocessing_config, memory=None):
        """Create transformers list for imputation and transforms."""
        transformers = []

        if "imputation" in preprocessing_config:
            imputation_config = preprocessing_config["imputation"]
            transform_config = preprocessing_config.get("transforms", {})

            self.logger.info(
                f"Creating transformers for {len(self.feature_columns)} feature groups"
            )
            # Loop through all user-defined feature groups and create pipelines
            for feature_group in tqdm(
                self.feature_columns.keys(),
                desc="Creating feature pipelines",
                unit="group",
            ):
                pipeline_info = self._create_feature_pipeline(
                    feature_group, imputation_config, transform_config, memory
                )
                if pipeline_info:
                    transformers.append(pipeline_info)
                    self.logger.debug(f"Added transformer for {feature_group}")
                else:
                    self.logger.debug(f"Skipped transformer for {feature_group}")

        self.logger.info(f"Created {len(transformers)} transformers")
        return transformers

    def _create_feature_selection_step(
        self, preprocessing_config, base_transformer, memory=None
    ):
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
            k = params.get("k", DEFAULT_K_BEST)
            score_func_name = params.get("score_func")
            if not score_func_name:
                score_func_name = "f_regression" if self.is_regression else "f_classif"

            # Map score function names to actual functions
            score_func = SCORE_FUNCS.get(score_func_name, SCORE_FUNCS["f_classif"])

            # Create SelectKBest selector
            selector = SelectKBest(score_func=score_func, k=k)
            pipeline_steps.append(("feature_selection", selector))

        elif method == "variance_threshold":
            # Get parameters for VarianceThreshold
            params = feature_select_config.get("params", {})
            threshold = params.get("threshold", DEFAULT_VARIANCE_THRESHOLD)

            # Create VarianceThreshold selector
            selector = VarianceThreshold(threshold=threshold)
            pipeline_steps.append(("feature_selection", selector))

        elif method == "rfe":
            # Get parameters for RFE
            params = feature_select_config.get("params", {})
            n_features = params.get("n_features", DEFAULT_RFE_N_FEATURES)
            step = params.get("step", DEFAULT_RFE_STEP)
            estimator_name = params.get("estimator", "logistic_regression")

            # Create estimator for RFE
            if estimator_name == "logistic_regression":
                estimator = LogisticRegression(
                    max_iter=DEFAULT_LOGISTIC_REGRESSION_MAX_ITER
                )
            else:
                # Default to logistic regression
                estimator = LogisticRegression(
                    max_iter=DEFAULT_LOGISTIC_REGRESSION_MAX_ITER
                )

            # Create RFE selector
            selector = RFE(
                estimator=estimator, n_features_to_select=n_features, step=step
            )
            pipeline_steps.append(("feature_selection", selector))

        # Create the full pipeline
        return Pipeline(steps=pipeline_steps, memory=memory)

    def create_pipeline(self):
        """
        Create the preprocessing pipeline based on config.

        Returns:
            Pipeline, ColumnTransformer, or FunctionTransformer: The configured preprocessing pipeline.
        """
        self.logger.info("Creating preprocessing pipeline...")
        preprocessing_config = self.config.get("preprocessing", {})

        # Get cache configuration
        memory = self._get_cache_config(preprocessing_config)

        # Create imputation and transform transformers
        transformers = self._create_imputation_transformers(
            preprocessing_config, memory
        )

        # Create base transformer (ColumnTransformer or passthrough)
        if transformers:
            base_transformer = ColumnTransformer(
                transformers=transformers, remainder="drop"
            )
            self.logger.info("Created ColumnTransformer with transformers")
        else:
            base_transformer = FunctionTransformer(func=None)
            self.logger.info("No transformers configured, using passthrough")

        # Add feature selection if specified
        final_pipeline = self._create_feature_selection_step(
            preprocessing_config, base_transformer, memory
        )

        if final_pipeline != base_transformer:
            self.logger.info("Added feature selection step to pipeline")

        # Wrap with DataFramePreservingTransformer to maintain DataFrame structure
        final_pipeline = DataFramePreservingTransformer(final_pipeline)
        self.logger.info("Wrapped pipeline with DataFramePreservingTransformer")

        self.logger.info("Preprocessing pipeline creation completed")
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
        print(f"Pipeline transformers: {len(pipeline.transformers_)}")
        for name, transformer, columns in pipeline.transformers_:
            print(f"  - {name}: {type(transformer).__name__} on {columns}")
    else:
        print("Pipeline: No-op transformer")

    print("Preprocessing pipeline ready for use!")
    print("To use: pipeline.fit_transform(X_train)")
