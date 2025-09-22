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

        # Handle imputation
        if "imputation" in preprocessing_config:
            imputation_config = preprocessing_config["imputation"]

            # Create imputers for each feature type
            for feature_type, config in imputation_config.items():
                if (
                    feature_type in self.feature_columns
                    and self.feature_columns[feature_type]
                ):
                    cols = self.feature_columns[feature_type]
                    method = config["method"]

                    if method == "constant":
                        fill_value = config.get("fill_value", "missing")
                        imputer = SimpleImputer(
                            strategy="constant", fill_value=fill_value
                        )
                    else:
                        imputer = SimpleImputer(strategy=method)

                    transformers.append((f"{feature_type}_imputer", imputer, cols))

        # Handle transforms
        if "transforms" in preprocessing_config:
            transform_config = preprocessing_config["transforms"]

            # Float transformations
            if "float" in transform_config and self.feature_columns["float"]:
                method = transform_config["float"]["method"]
                cols = self.feature_columns["float"]

                if method == "standard_scaler":
                    scaler = StandardScaler()
                elif method == "min_max_scaler":
                    scaler = MinMaxScaler()
                elif method == "robust_scaler":
                    scaler = RobustScaler()
                else:
                    scaler = FunctionTransformer(lambda x: x)  # No-op

                transformers.append((f"float_{method}", scaler, cols))

            # Categorical transformations
            if (
                "categorical" in transform_config
                and self.feature_columns["categorical"]
            ):
                method = transform_config["categorical"]["method"]
                cols = self.feature_columns["categorical"]

                if method == "one_hot_encoding":
                    encoder = OneHotEncoder(sparse_output=False, drop="first")
                    transformers.append((f"categorical_{method}", encoder, cols))
                elif method == "label_encoding":
                    # Label encoding needs special handling since it returns 1D
                    encoder = LabelEncoder()
                    # We'll handle this separately as it needs different treatment
                    pass
                else:
                    # No-op for other methods
                    pass

        # Create ColumnTransformer
        if transformers:
            self.pipeline = ColumnTransformer(
                transformers=transformers,
                remainder="passthrough",  # Keep other columns as-is
            )
        else:
            # No transformations specified
            self.pipeline = FunctionTransformer(lambda x: x)

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

    if isinstance(pipeline, ColumnTransformer):
        print(f"Pipeline steps: {len(pipeline.transformers)}")
        for name, transformer, columns in pipeline.transformers:
            print(f"  - {name}: {type(transformer).__name__} on {columns}")
    else:
        print("Pipeline: No-op transformer")

    print("Preprocessing pipeline ready for use!")
    print("To use: pipeline.fit_transform(X_train)")
