import pandas as pd
import yaml
import logging


class DataLoader:
    """Simple data loader for YAML-configured datasets."""

    def __init__(self, config_path):
        """Initialize with YAML config file path."""
        self.logger = logging.getLogger(__name__)
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)
        self.logger.info(f"DataLoader initialized with config: {config_path}")

    def load_data(self):
        """Load data according to config.

        Returns:
            dict: Dictionary with keys 'train', 'test', 'validation' containing DataFrames
        """
        self.logger.info("Starting data loading process...")

        # Load data files
        self.logger.info(
            f"Loading train data from: {self.config['files']['train_data']}"
        )
        train_df = pd.read_csv(self.config["files"]["train_data"])
        self.logger.info(f"Train data loaded: {train_df.shape}")

        self.logger.info(f"Loading test data from: {self.config['files']['test_data']}")
        test_df = pd.read_csv(self.config["files"]["test_data"])
        self.logger.info(f"Test data loaded: {test_df.shape}")

        validation_df = None
        if self.config["files"]["validation_data"] is not None:
            self.logger.info(
                f"Loading validation data from: {self.config['files']['validation_data']}"
            )
            validation_df = pd.read_csv(self.config["files"]["validation_data"])
            self.logger.info(f"Validation data loaded: {validation_df.shape}")
        else:
            self.logger.info("No validation data specified")

        # Get columns to keep based on config
        columns_to_keep = self._get_columns_to_keep()
        self.logger.info(f"Columns to keep: {columns_to_keep}")

        # Filter DataFrames to only include specified columns that exist
        self.logger.info("Filtering train data columns...")
        train_df = self._filter_columns(train_df, columns_to_keep)
        self.logger.info(f"Train data after filtering: {train_df.shape}")

        self.logger.info("Filtering test data columns...")
        test_df = self._filter_columns(test_df, columns_to_keep)
        self.logger.info(f"Test data after filtering: {test_df.shape}")

        if validation_df is not None:
            self.logger.info("Filtering validation data columns...")
            validation_df = self._filter_columns(validation_df, columns_to_keep)
            self.logger.info(f"Validation data after filtering: {validation_df.shape}")

        self.logger.info("Data loading completed successfully")
        return {"train": train_df, "test": test_df, "validation": validation_df}

    def _filter_columns(self, df, columns_to_keep):
        """Filter DataFrame to only include columns that exist in the DataFrame."""
        existing_columns = [col for col in columns_to_keep if col in df.columns]
        missing_columns = [col for col in columns_to_keep if col not in df.columns]
        if missing_columns:
            self.logger.warning(f"Missing columns in DataFrame: {missing_columns}")
        filtered_df = df[existing_columns]
        self.logger.debug(
            f"Filtered DataFrame from {df.shape[1]} to {filtered_df.shape[1]} columns"
        )
        return filtered_df

    def _get_columns_to_keep(self):
        """Get list of columns that should be loaded based on config."""
        columns = set()
        self.logger.debug("Building column list from config...")

        # Add ID column
        if "id" in self.config["data"]:
            columns.add(self.config["data"]["id"]["column"])
            self.logger.debug(f"Added ID column: {self.config['data']['id']['column']}")

        # Add target column
        if "target" in self.config["data"]:
            columns.add(self.config["data"]["target"]["column"])
            self.logger.debug(
                f"Added target column: {self.config['data']['target']['column']}"
            )

        # Add feature columns from all user-defined feature groups
        if "features" in self.config["data"]:
            features = self.config["data"]["features"]
            for feature_group, feature_list in features.items():
                if isinstance(feature_list, list):
                    columns.update(feature_list)
                    self.logger.debug(
                        f"Added {len(feature_list)} columns from feature group '{feature_group}'"
                    )

        self.logger.info(f"Total columns to keep: {len(columns)}")
        return list(columns)


def load_dataset(config_path):
    """Convenience function to load dataset from config.

    Args:
        config_path (str): Path to YAML config file

    Returns:
        dict: Dictionary with keys 'train', 'test', 'validation' containing DataFrames
    """
    loader = DataLoader(config_path)
    return loader.load_data()


# Test the setup
if __name__ == "__main__":
    # Test with titanic config
    config_path = "ds_nailgun/configs/examples/titanic_data_config.yaml"
    data = load_dataset(config_path)

    print("Train shape:", data["train"].shape)
    print("Test shape:", data["test"].shape)
    print("Validation is None:", data["validation"] is None)
    print("Train columns:", list(data["train"].columns)[:10])  # Show first 10 columns
