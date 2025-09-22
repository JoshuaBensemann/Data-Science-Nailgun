"""
Data Science Experiment Controller

Orchestrates data science experiments by coordinating
multiple modules and configuration files.
"""

import yaml
from typing import Dict
from .dataloader import DataLoader
from .preprocessing import create_preprocessing_pipeline


class ExperimentController:
    """Main controller for running data science experiments."""

    def __init__(self, config_paths: Dict[str, str]):
        """
        Initialize controller with configuration file paths.

        Args:
            config_paths: Dictionary mapping config types to file paths
                e.g., {'data': 'configs/examples/titanic_data_config.yaml'}
        """
        self.config_paths = config_paths
        self.configs = {}
        self.data = None
        self.preprocessing_pipeline = None
        self.experiment_state = {}

    def load_configs(self):
        """Load all configuration files."""
        for config_type, config_path in self.config_paths.items():
            with open(config_path, "r") as file:
                self.configs[config_type] = yaml.safe_load(file)
            print(f"Loaded {config_type} config from {config_path}")

    def setup_data(self):
        """Load and prepare the dataset using data_loader."""
        if "data" not in self.config_paths:
            raise ValueError(
                "Data config path not found. Please provide a 'data' config path."
            )

        data_config_path = self.config_paths["data"]
        loader = DataLoader(data_config_path)
        self.data = loader.load_data()

        print("Data loaded successfully:")
        print(f"  Train: {self.data['train'].shape}")
        print(f"  Test: {self.data['test'].shape}")
        print(
            f"  Validation: {'None' if self.data['validation'] is None else self.data['validation'].shape}"
        )

        return self.data

    def setup_preprocessing(self):
        """Create preprocessing pipeline (ready for use in larger pipeline)."""
        if "data" not in self.config_paths:
            raise ValueError("Data config path not found for preprocessing.")

        # Create preprocessing pipeline
        data_config_path = self.config_paths["data"]
        preprocessor = create_preprocessing_pipeline(data_config_path)
        self.preprocessing_pipeline = preprocessor.create_pipeline()

        print("Preprocessing pipeline created successfully:")
        print(f"  Pipeline type: {type(self.preprocessing_pipeline).__name__}")
        print("  Ready for use in larger pipeline")

        return self.preprocessing_pipeline

    def run_experiment(self):
        """Run the complete experiment pipeline."""
        print("Starting Data Science Experiment...")

        # Step 1: Load configurations
        self.load_configs()

        # Step 2: Setup data
        self.setup_data()

        # Step 3: Preprocessing (placeholder)
        self.setup_preprocessing()

        # Future steps can be added here:
        # - Model training
        # - Evaluation
        # - Reporting

        print("Experiment setup complete!")
        return self.experiment_state

    def get_data(self):
        """Get the loaded data."""
        return self.data

    def get_preprocessing_pipeline(self):
        """Get the preprocessing pipeline."""
        return self.preprocessing_pipeline

    def get_config(self, config_type: str):
        """Get a specific configuration."""
        return self.configs.get(config_type)

    def get_experiment_state(self):
        """Get the current experiment state."""
        return self.experiment_state


def run_experiment(config_paths: Dict[str, str]) -> ExperimentController:
    """
    Convenience function to run a complete experiment.

    Args:
        config_paths: Dictionary mapping config types to file paths

    Returns:
        ExperimentController: The controller instance with loaded data and configs
    """
    controller = ExperimentController(config_paths)
    controller.run_experiment()
    return controller


# Example usage
if __name__ == "__main__":
    # Example configuration paths
    config_paths = {
        "data": "configs/examples/titanic_data_config.yaml",
        # Future configs can be added:
        # 'model': 'configs/model_config.yaml',
        # 'preprocessing': 'configs/preprocessing_config.yaml',
    }

    # Run experiment
    controller = run_experiment(config_paths)

    # Access loaded data
    data = controller.get_data()
    if data:
        print("\nData is ready for use:")
        print(f"Train shape: {data['train'].shape}")
        print(f"Test shape: {data['test'].shape}")
    else:
        print("\nData not loaded yet. Call setup_data() first.")
