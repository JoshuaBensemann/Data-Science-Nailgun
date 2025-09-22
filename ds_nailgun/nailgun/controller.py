"""
Data Science Experiment Controller

Orchestrates data science experiments by coordinating
multiple modules and configuration files.
"""

import yaml
from typing import Dict, List, Optional
from sklearn.pipeline import Pipeline
from .dataloader import DataLoader
from .preprocessing import create_preprocessing_pipeline
from .model_factory import create_model


class ExperimentController:
    """Main controller for running data science experiments."""

    def __init__(self, config_paths: Dict[str, str], model_config_paths: Optional[List[str]] = None):
        """
        Initialize controller with configuration file paths.

        Args:
            config_paths: Dictionary mapping config types to file paths
                e.g., {'data': 'configs/examples/titanic_data_config.yaml'}
            model_config_paths: List of model configuration file paths
                e.g., ['configs/model_presets/random_forest_classifier_config.yaml']
        """
        self.config_paths = config_paths
        self.model_config_paths = model_config_paths or []
        self.configs = {}
        self.data = None
        self.preprocessing_pipeline = None
        self.models = []
        self.trained_pipelines = []
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

    def setup_models(self):
        """Create model instances from model configuration files."""
        if not self.model_config_paths:
            print("No model configs provided.")
            return

        for config_path in self.model_config_paths:
            model = create_model(config_path)
            self.models.append(model)
            print(f"Created model from {config_path}: {type(model).__name__}")

        return self.models

    def train_models(self):
        """Train models by creating pipelines and fitting on training data."""
        if not self.models:
            print("No models to train. Call setup_models() first.")
            return

        if not self.data or 'train' not in self.data:
            raise ValueError("Data not loaded. Call setup_data() first.")

        train_data = self.data['train']
        X_train = train_data.drop(columns=[self.configs['data']['data']['target']['column']])
        y_train = train_data[self.configs['data']['data']['target']['column']]

        for i, model in enumerate(self.models):
            # Create pipeline: preprocessing + model
            pipeline = Pipeline([
                ('preprocessing', self.preprocessing_pipeline),
                ('model', model)
            ])

            # Fit the pipeline
            print(f"Training model {i+1}/{len(self.models)}: {type(model).__name__}")
            pipeline.fit(X_train, y_train)
            self.trained_pipelines.append(pipeline)
            print("  Model trained successfully")

        return self.trained_pipelines

    def run_experiment(self):
        """Run the complete experiment pipeline."""
        print("Starting Data Science Experiment...")

        # Step 1: Load configurations
        self.load_configs()

        # Step 2: Setup data
        self.setup_data()

        # Step 3: Setup preprocessing
        self.setup_preprocessing()

        # Step 4: Setup models
        self.setup_models()

        # Step 5: Train models
        self.train_models()

        print("Experiment complete!")
        return self.experiment_state

    def get_data(self):
        """Get the loaded data."""
        return self.data

    def get_preprocessing_pipeline(self):
        """Get the preprocessing pipeline."""
        return self.preprocessing_pipeline

    def get_models(self):
        """Get the list of model instances."""
        return self.models

    def get_trained_pipelines(self):
        """Get the list of trained pipelines."""
        return self.trained_pipelines

    def get_config(self, config_type: str):
        """Get a specific configuration."""
        return self.configs.get(config_type)

    def get_experiment_state(self):
        """Get the current experiment state."""
        return self.experiment_state


def run_experiment(config_paths: Dict[str, str], model_config_paths: Optional[List[str]] = None) -> ExperimentController:
    """
    Convenience function to run a complete experiment.

    Args:
        config_paths: Dictionary mapping config types to file paths
        model_config_paths: List of model configuration file paths

    Returns:
        ExperimentController: The controller instance with loaded data and configs
    """
    controller = ExperimentController(config_paths, model_config_paths)
    controller.run_experiment()
    return controller


# Example usage
if __name__ == "__main__":
    # Example configuration paths
    config_paths = {
        "data": "ds_nailgun/configs/examples/titanic_data_config.yaml",
        # Future configs can be added:
        # 'model': 'configs/model_config.yaml',
        # 'preprocessing': 'configs/preprocessing_config.yaml',
    }

    # Example model configuration paths
    model_config_paths = [
        "ds_nailgun/configs/model_presets/random_forest_classifier_config.yaml",
        "ds_nailgun/configs/model_presets/xgboost_classifier_config.yaml",
    ]

    # Run experiment
    controller = run_experiment(config_paths, model_config_paths)

    # Access trained pipelines
    pipelines = controller.get_trained_pipelines()
    if pipelines:
        print(f"\nTrained {len(pipelines)} model pipelines:")
        for i, pipeline in enumerate(pipelines):
            print(f"  Pipeline {i+1}: {type(pipeline.named_steps['model']).__name__}")
    else:
        print("\nNo pipelines trained yet.")
