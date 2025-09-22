import pandas as pd
import yaml


class DataLoader:
    """Simple data loader for YAML-configured datasets."""

    def __init__(self, config_path):
        """Initialize with YAML config file path."""
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

    def load_data(self):
        """Load data according to config."""
        # Load data files
        train_df = pd.read_csv(self.config["files"]["train_data"])
        test_df = pd.read_csv(self.config["files"]["test_data"])

        validation_df = None
        if self.config["files"]["validation_data"] is not None:
            validation_df = pd.read_csv(self.config["files"]["validation_data"])

        # Get columns to keep based on config
        columns_to_keep = self._get_columns_to_keep()

        # Filter DataFrames to only include specified columns that exist
        train_df = self._filter_columns(train_df, columns_to_keep)
        test_df = self._filter_columns(test_df, columns_to_keep)
        if validation_df is not None:
            validation_df = self._filter_columns(validation_df, columns_to_keep)

        return train_df, test_df, validation_df

    def _filter_columns(self, df, columns_to_keep):
        """Filter DataFrame to only include columns that exist in the DataFrame."""
        existing_columns = [col for col in columns_to_keep if col in df.columns]
        return df[existing_columns]

    def _get_columns_to_keep(self):
        """Get list of columns that should be loaded based on config."""
        columns = set()

        # Add ID column
        if "id" in self.config["data"]:
            columns.add(self.config["data"]["id"]["column"])

        # Add target column
        if "target" in self.config["data"]:
            columns.add(self.config["data"]["target"]["column"])

        # Add feature columns
        if "features" in self.config["data"]:
            features = self.config["data"]["features"]
            for feature_list in features.values():
                if isinstance(feature_list, list):
                    columns.update(feature_list)

        return list(columns)


def load_dataset(config_path):
    """Convenience function to load dataset from config."""
    loader = DataLoader(config_path)
    return loader.load_data()


# Test the setup
if __name__ == "__main__":
    # Test with titanic config
    config_path = "configs/examples/titanic_data_config.yaml"
    train_df, test_df, val_df = load_dataset(config_path)

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
    print("Validation:", val_df is None)
    print("Train columns:", list(train_df.columns)[:10])  # Show first 10 columns
