"""
Data Science Experiment Controller

Orchestrates data science experiments by coordinating
multiple modules and configuration files.
"""

import yaml
import logging
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from .dataloader import DataLoader
from .preprocessing import create_preprocessing_pipeline
from .model_factory import create_model


class ExperimentController:
    """Main controller for running data science experiments."""

    def __init__(self, experiment_config_path: str):
        """
        Initialize controller with experiment configuration file path.

        Args:
            experiment_config_path: Path to the experiment configuration YAML file
        """
        self.experiment_config_path = experiment_config_path

        # Load experiment config
        with open(experiment_config_path, "r") as f:
            self.experiment_config = yaml.safe_load(f)

        # Extract configurations
        self.config_paths = {
            "data": self.experiment_config["data"]["config_path"],
        }
        self.model_config_paths = self.experiment_config["models"]["config_paths"]
        self.output_config = self.experiment_config.get("output", {})

        # Setup logging
        self.setup_logging()

        # Initialize other attributes
        self.configs = {}
        self.data = None
        self.preprocessing_pipeline = None
        self.models = []
        self.trained_pipelines = []
        self.experiment_state = {}

    def setup_logging(self):
        """Setup logging based on experiment configuration."""
        logging_config = self.experiment_config.get("logging", {})
        level = getattr(logging, logging_config.get("level", "INFO").upper())
        format_str = logging_config.get(
            "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        logging.basicConfig(
            level=level,
            format=format_str,
            handlers=[
                logging.StreamHandler(),  # Console handler
            ],
        )

        # Add file handler if specified
        if "file" in logging_config:
            log_file = logging_config["file"]
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(format_str))
            logging.getLogger().addHandler(file_handler)

        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"Initialized logging for experiment: {self.experiment_config['experiment']['name']}"
        )

    def load_configs(self):
        """Load all configuration files."""
        for config_type, config_path in self.config_paths.items():
            with open(config_path, "r") as file:
                self.configs[config_type] = yaml.safe_load(file)
            self.logger.info(f"Loaded {config_type} config from {config_path}")

    def setup_data(self):
        """Load and prepare the dataset using data_loader."""
        if "data" not in self.config_paths:
            raise ValueError(
                "Data config path not found. Please provide a 'data' config path."
            )

        data_config_path = self.config_paths["data"]
        loader = DataLoader(data_config_path)
        self.data = loader.load_data()

        self.logger.info("Data loaded successfully:")
        self.logger.info(f"  Train: {self.data['train'].shape}")
        self.logger.info(f"  Test: {self.data['test'].shape}")
        self.logger.info(
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

        self.logger.info("Preprocessing pipeline created successfully:")
        self.logger.info(
            f"  Pipeline type: {type(self.preprocessing_pipeline).__name__}"
        )
        self.logger.info("  Ready for use in larger pipeline")

        return self.preprocessing_pipeline

    def setup_models(self):
        """Create model instances from model configuration files."""
        if not self.model_config_paths:
            self.logger.info("No model configs provided.")
            return

        for config_path in self.model_config_paths:
            model = create_model(config_path)
            self.models.append(model)
            self.logger.info(
                f"Created model from {config_path}: {type(model).__name__}"
            )

        return self.models

    def train_models(self):
        """Train models by creating pipelines and fitting on training data."""
        if not self.models:
            self.logger.warning("No models to train. Call setup_models() first.")
            return

        if not self.data or "train" not in self.data:
            raise ValueError("Data not loaded. Call setup_data() first.")

        train_data = self.data["train"]
        X_train = train_data.drop(
            columns=[self.configs["data"]["data"]["target"]["column"]]
        )
        y_train = train_data[self.configs["data"]["data"]["target"]["column"]]

        for i, (model, config_path) in enumerate(
            zip(self.models, self.model_config_paths)
        ):
            # Load the full config to check for hypertuning
            with open(config_path, "r") as f:
                full_config = yaml.safe_load(f)

            estimator = model
            if (
                "hypertuning" in full_config
                and full_config["hypertuning"]["method"] == "grid_search"
            ):
                hypertuning_config = full_config["hypertuning"]
                estimator = GridSearchCV(
                    model,
                    param_grid=hypertuning_config["parameters"],
                    cv=hypertuning_config.get("cv", 5),
                    scoring=hypertuning_config.get("scoring", "accuracy"),
                    verbose=1,
                )
                self.logger.info(
                    f"Training model {i + 1}/{len(self.models)}: {type(model).__name__} with Grid Search"
                )
            else:
                self.logger.info(
                    f"Training model {i + 1}/{len(self.models)}: {type(model).__name__}"
                )

            # Create pipeline: preprocessing + estimator
            pipeline = Pipeline(
                [("preprocessing", self.preprocessing_pipeline), ("model", estimator)]
            )

            # Fit the pipeline
            pipeline.fit(X_train, y_train)
            self.trained_pipelines.append(pipeline)

            # Print best parameters if grid search was used
            if isinstance(estimator, GridSearchCV):
                self.logger.info(f"  Best parameters: {estimator.best_params_}")
                self.logger.info(
                    f"  Best cross-validation score: {estimator.best_score_:.4f}"
                )

            self.logger.info("  Model trained successfully")

        return self.trained_pipelines

    def save_models(self, save_directory: str, save_format: str = "joblib"):
        """Save trained model pipelines to disk."""
        if not self.trained_pipelines:
            self.logger.warning("No trained pipelines to save.")
            return

        os.makedirs(save_directory, exist_ok=True)

        for i, pipeline in enumerate(self.trained_pipelines):
            model_name = type(pipeline.named_steps["model"]).__name__
            if hasattr(pipeline.named_steps["model"], "best_estimator_"):
                model_name = type(
                    pipeline.named_steps["model"].best_estimator_
                ).__name__

            filename = f"pipeline_{i + 1}_{model_name.lower()}.{save_format}"
            filepath = os.path.join(save_directory, filename)

            if save_format == "joblib":
                joblib.dump(pipeline, filepath)
            elif save_format == "pickle":
                import pickle

                with open(filepath, "wb") as f:
                    pickle.dump(pipeline, f)

            self.logger.info(f"Saved pipeline {i + 1} to {filepath}")

    def run_experiment(self):
        """Run the complete experiment pipeline."""
        self.logger.info("Starting Data Science Experiment...")

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

        self.logger.info("Experiment complete!")
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


def run_experiment(experiment_config_path: str) -> ExperimentController:
    """
    Convenience function to run a complete experiment.

    Args:
        experiment_config_path: Path to the experiment configuration YAML file

    Returns:
        ExperimentController: The controller instance with loaded data and configs
    """
    controller = ExperimentController(experiment_config_path)
    controller.run_experiment()
    return controller


# Example usage
if __name__ == "__main__":
    # Experiment configuration path
    experiment_config_path = (
        "ds_nailgun/configs/examples/titanic_experiment_config.yaml"
    )

    print(f"Running experiment from config: {experiment_config_path}")

    # Run experiment
    controller = run_experiment(experiment_config_path)

    # Save trained pipelines (if output config exists)
    if controller.output_config:
        save_directory = controller.output_config["model_save_directory"]
        save_format = controller.output_config.get("save_format", "joblib")
        controller.save_models(save_directory, save_format)

    # Access trained pipelines
    pipelines = controller.get_trained_pipelines()
    if pipelines:
        print(f"\nTrained {len(pipelines)} model pipelines:")
        for i, pipeline in enumerate(pipelines):
            model_name = type(pipeline.named_steps["model"]).__name__
            if hasattr(pipeline.named_steps["model"], "best_estimator_"):
                model_name = f"GridSearchCV({type(pipeline.named_steps['model'].best_estimator_).__name__})"
            print(f"  Pipeline {i + 1}: {model_name}")
    else:
        print("\nNo pipelines trained yet.")
