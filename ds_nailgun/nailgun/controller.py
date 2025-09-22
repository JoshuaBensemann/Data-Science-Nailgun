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
            "data": self.experiment_config["data"]["config_paths"],  # Now a list
        }
        self.model_config_paths = self.experiment_config["models"]["config_paths"]
        self.output_config = self.experiment_config.get("output", {})

        # Setup logging
        self.setup_logging()

        # Initialize other attributes
        self.configs = {}
        self.data = {}  # Now a dict with data config names as keys
        self.preprocessing_pipeline = {}  # Now a dict with data config names as keys
        self.models = []
        self.trained_pipelines = {}  # Now a dict with experiment names as keys
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
        # Load data configs (now multiple)
        self.configs["data"] = []
        for i, data_config_path in enumerate(self.config_paths["data"]):
            with open(data_config_path, "r") as file:
                config = yaml.safe_load(file)
                self.configs["data"].append(
                    {
                        "path": data_config_path,
                        "config": config,
                        "name": f"data_config_{i + 1}",
                    }
                )
            self.logger.info(f"Loaded data config {i + 1} from {data_config_path}")

        # Load model configs
        self.configs["models"] = []
        for model_config_path in self.model_config_paths:
            with open(model_config_path, "r") as file:
                config = yaml.safe_load(file)
                self.configs["models"].append(
                    {"path": model_config_path, "config": config}
                )
            self.logger.info(f"Loaded model config from {model_config_path}")

    def setup_data(self):
        """Load and prepare datasets for all data configurations."""
        if "data" not in self.configs or not self.configs["data"]:
            raise ValueError("Data configs not loaded. Call load_configs() first.")

        for data_config_info in self.configs["data"]:
            data_config_path = data_config_info["path"]
            data_config_name = data_config_info["name"]

            loader = DataLoader(data_config_path)
            self.data[data_config_name] = loader.load_data()

            self.logger.info(f"Data loaded for {data_config_name}:")
            self.logger.info(f"  Train: {self.data[data_config_name]['train'].shape}")
            self.logger.info(f"  Test: {self.data[data_config_name]['test'].shape}")
            self.logger.info(
                f"  Validation: {'None' if self.data[data_config_name]['validation'] is None else self.data[data_config_name]['validation'].shape}"
            )

        return self.data

    def setup_preprocessing(self):
        """Create preprocessing pipelines for all data configurations."""
        if "data" not in self.configs or not self.configs["data"]:
            raise ValueError("Data configs not loaded. Call load_configs() first.")

        for data_config_info in self.configs["data"]:
            data_config_path = data_config_info["path"]
            data_config_name = data_config_info["name"]

            # Create preprocessing pipeline
            preprocessor = create_preprocessing_pipeline(data_config_path)
            self.preprocessing_pipeline[data_config_name] = (
                preprocessor.create_pipeline()
            )

            self.logger.info(f"Preprocessing pipeline created for {data_config_name}:")
            self.logger.info(
                f"  Pipeline type: {type(self.preprocessing_pipeline[data_config_name]).__name__}"
            )

        self.logger.info("All preprocessing pipelines ready for use")
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
        """Train models by creating pipelines for each data config and model combination."""
        if not self.models:
            self.logger.warning("No models to train. Call setup_models() first.")
            return

        if not self.data:
            raise ValueError("Data not loaded. Call setup_data() first.")

        if not self.preprocessing_pipeline:
            raise ValueError(
                "Preprocessing pipelines not created. Call setup_preprocessing() first."
            )

        experiment_count = 0

        # Train models for each combination of data config and model config
        for data_config_info in self.configs["data"]:
            data_config_name = data_config_info["name"]
            data_config = data_config_info["config"]
            train_data = self.data[data_config_name]["train"]

            X_train = train_data.drop(columns=[data_config["data"]["target"]["column"]])
            y_train = train_data[data_config["data"]["target"]["column"]]

            for model_idx, (model, model_config_info) in enumerate(
                zip(self.models, self.configs["models"])
            ):
                experiment_count += 1
                experiment_name = f"{data_config_name}_model_{model_idx + 1}"

                # Load the full model config to check for hypertuning
                full_config = model_config_info["config"]

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
                        f"Training experiment {experiment_count}: {data_config_name} + {type(model).__name__} with Grid Search"
                    )
                else:
                    self.logger.info(
                        f"Training experiment {experiment_count}: {data_config_name} + {type(model).__name__}"
                    )

                # Create pipeline: preprocessing + estimator
                pipeline = Pipeline(
                    [
                        (
                            "preprocessing",
                            self.preprocessing_pipeline[data_config_name],
                        ),
                        ("model", estimator),
                    ]
                )

                # Fit the pipeline
                pipeline.fit(X_train, y_train)

                # Store the trained pipeline
                self.trained_pipelines[experiment_name] = {
                    "pipeline": pipeline,
                    "data_config": data_config_name,
                    "model_config": model_config_info["path"],
                    "model_name": type(model).__name__,
                }

                # Print best parameters if grid search was used
                if isinstance(estimator, GridSearchCV):
                    self.logger.info(f"  Best parameters: {estimator.best_params_}")
                    self.logger.info(
                        f"  Best cross-validation score: {estimator.best_score_:.4f}"
                    )

                self.logger.info(f"  Experiment {experiment_name} trained successfully")

        self.logger.info(f"Completed training {experiment_count} model experiments")
        return self.trained_pipelines

    def save_models(self, save_directory: str, save_format: str = "joblib"):
        """Save trained model pipelines to disk."""
        if not self.trained_pipelines:
            self.logger.warning("No trained pipelines to save.")
            return

        os.makedirs(save_directory, exist_ok=True)

        for experiment_name, experiment_info in self.trained_pipelines.items():
            pipeline = experiment_info["pipeline"]
            model_name = experiment_info["model_name"]

            # Get the actual model name (handle GridSearchCV)
            if hasattr(pipeline.named_steps["model"], "best_estimator_"):
                actual_model_name = type(
                    pipeline.named_steps["model"].best_estimator_
                ).__name__
            else:
                actual_model_name = model_name

            filename = (
                f"pipeline_{experiment_name}_{actual_model_name.lower()}.{save_format}"
            )
            filepath = os.path.join(save_directory, filename)

            if save_format == "joblib":
                joblib.dump(pipeline, filepath)
            elif save_format == "pickle":
                import pickle

                with open(filepath, "wb") as f:
                    pickle.dump(pipeline, f)

            self.logger.info(f"Saved experiment {experiment_name} to {filepath}")

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
        print(f"\nTrained {len(pipelines)} model experiments:")
        for experiment_name, experiment_info in pipelines.items():
            pipeline = experiment_info["pipeline"]
            data_config = experiment_info["data_config"]
            model_name = experiment_info["model_name"]

            # Get the actual model name (handle GridSearchCV)
            if hasattr(pipeline.named_steps["model"], "best_estimator_"):
                display_name = f"GridSearchCV({type(pipeline.named_steps['model'].best_estimator_).__name__})"
            else:
                display_name = model_name

            print(f"  Experiment {experiment_name}: {data_config} + {display_name}")
    else:
        print("\nNo pipelines trained yet.")
