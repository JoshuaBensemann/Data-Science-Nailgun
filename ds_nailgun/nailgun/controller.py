"""
Data Science Experiment Controller

Orchestrates data science experiments by coordinating
multiple modules and configuration files.
"""

import yaml
import logging
import joblib
import os
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_pinball_loss
import pandas as pd
from .dataloader import DataLoader
from .preprocessing import create_preprocessing_pipeline
from .model_factory import create_model
from .consts import HYPERTUNING_METHODS, SCORING_NAMES, SAVE_FORMATS
from tqdm import tqdm


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

        # Setup output directory structure
        self.setup_output_directory()

        # Setup logging (now that output directory exists)
        self.setup_logging()

        # Initialize other attributes
        self.configs = {}
        self.data = {}  # Now a dict with data config names as keys
        self.preprocessing_pipeline = {}  # Now a dict with data config names as keys
        self.models = []
        self.trained_pipelines = {}  # Now a dict with experiment names as keys
        self.experiment_state = {}

    def setup_output_directory(self):
        """Create timestamped output directory structure."""
        if not self.output_config:
            self.output_base_dir = None
            self.output_dir = None
            return

        # Create base directory if it doesn't exist
        base_dir = self.output_config.get("base_directory", "experiments")
        os.makedirs(base_dir, exist_ok=True)

        # Create timestamped experiment directory
        experiment_name = (
            self.experiment_config["experiment"]["name"].replace(" ", "_").lower()
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")

        # Create subdirectories
        self.models_dir = os.path.join(self.output_dir, "models")
        self.results_dir = os.path.join(self.output_dir, "results")
        self.logs_dir = os.path.join(self.output_dir, "logs")

        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        self.output_base_dir = base_dir

    def setup_logging(self):
        """Setup logging based on experiment configuration."""
        logging_config = self.experiment_config.get("logging", {})
        level = getattr(logging, logging_config.get("level", "INFO").upper())
        format_str = logging_config.get(
            "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Create logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level)

        # Clear any existing handlers
        self.logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(format_str))
        self.logger.addHandler(console_handler)

        # File handler (if output directory exists)
        if self.output_dir and "file" in logging_config:
            log_file = logging_config["file"]
            # Use the logs directory in the timestamped output folder
            log_filepath = os.path.join(self.logs_dir, os.path.basename(log_file))
            os.makedirs(os.path.dirname(log_filepath), exist_ok=True)

            file_handler = logging.FileHandler(log_filepath)
            file_handler.setFormatter(logging.Formatter(format_str))
            self.logger.addHandler(file_handler)

        self.logger.info(
            f"Initialized logging for experiment: {self.experiment_config['experiment']['name']}"
        )
        if self.output_dir:
            self.logger.info(f"Logs will be saved to: {self.logs_dir}")

    def load_configs(self):
        """Load all configuration files."""
        # Load data configs (now multiple)
        self.configs["data"] = []
        for i, data_config_path in enumerate(
            tqdm(self.config_paths["data"], desc="Loading data configs", unit="config")
        ):
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
        for model_config_path in tqdm(
            self.model_config_paths, desc="Loading model configs", unit="config"
        ):
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

        for data_config_info in tqdm(
            self.configs["data"], desc="Setting up data", unit="config"
        ):
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

        for data_config_info in tqdm(
            self.configs["data"], desc="Setting up preprocessing", unit="pipeline"
        ):
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

        for config_path in tqdm(
            self.model_config_paths, desc="Setting up models", unit="model"
        ):
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
        total_experiments = len(self.configs["data"]) * len(self.models)
        self.logger.info(f"Starting training of {total_experiments} experiments")

        # Train models for each combination of data config and model config
        with tqdm(
            total=total_experiments, desc="Training experiments", unit="experiment"
        ) as pbar:
            for data_config_info in self.configs["data"]:
                data_config_name = data_config_info["name"]
                data_config = data_config_info["config"]
                train_data = self.data[data_config_name]["train"]

                X_train = train_data.drop(
                    columns=[data_config["data"]["target"]["column"]]
                )
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
                        and full_config["hypertuning"]["method"] in HYPERTUNING_METHODS
                    ):
                        hypertuning_config = full_config["hypertuning"]

                        # Handle special scoring metrics
                        scoring_config = hypertuning_config.get(
                            "scoring", {"name": "accuracy"}
                        )
                        if isinstance(scoring_config, str):
                            # Backward compatibility: if scoring is still a string
                            scoring = scoring_config
                        else:
                            # New format: scoring is a dict with name and optional parameters
                            scoring_name = scoring_config.get("name", "accuracy")
                            if scoring_name in SCORING_NAMES:
                                alpha = scoring_config.get(
                                    "alpha", 0.5
                                )  # Default to median if not specified
                                scoring = make_scorer(
                                    mean_pinball_loss,
                                    alpha=alpha,
                                    greater_is_better=False,
                                )
                            else:
                                scoring = scoring_name

                        estimator = GridSearchCV(
                            model,
                            param_grid=hypertuning_config["parameters"],
                            cv=hypertuning_config.get("cv", 5),
                            scoring=scoring,
                            verbose=2,  # Increased verbosity
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
                    self.logger.info(f"Fitting pipeline for {experiment_name}...")
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

                    self.logger.info(
                        f"  Experiment {experiment_name} trained successfully"
                    )
                    pbar.update(1)

        self.logger.info(f"Completed training {experiment_count} model experiments")
        return self.trained_pipelines

    def save_models(self):
        """Save trained model pipelines to the timestamped output directory."""
        if not self.trained_pipelines or not self.output_dir:
            return

        save_format = self.output_config.get("save_format", "joblib")

        for experiment_name, experiment_info in tqdm(
            self.trained_pipelines.items(), desc="Saving models", unit="model"
        ):
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
            filepath = os.path.join(self.models_dir, filename)

            if save_format in SAVE_FORMATS:
                if save_format == "joblib":
                    joblib.dump(pipeline, filepath)
                elif save_format == "pickle":
                    import pickle

                    with open(filepath, "wb") as f:
                        pickle.dump(pipeline, f)

            self.logger.info(f"Saved model {experiment_name} to {filepath}")

    def save_hyperparameter_results(self):
        """Save detailed hyperparameter tuning results."""
        if not self.trained_pipelines or not self.output_dir:
            return

        for experiment_name, experiment_info in tqdm(
            self.trained_pipelines.items(),
            desc="Saving hyperparameter results",
            unit="result",
        ):
            pipeline = experiment_info["pipeline"]

            # Check if this pipeline used GridSearchCV
            if hasattr(pipeline.named_steps["model"], "cv_results_"):
                cv_results = pipeline.named_steps["model"].cv_results_

                # Save detailed results as CSV
                results_file = os.path.join(
                    self.results_dir, f"hypertuning_{experiment_name}.csv"
                )

                # Convert cv_results to DataFrame and save as CSV
                df_results = pd.DataFrame(cv_results)
                df_results.to_csv(results_file, index=False)

                self.logger.info(
                    f"Saved hyperparameter results for {experiment_name} to {results_file}"
                )

    def save_config_summary(self):
        """Save a summary of all configurations used in the experiment."""
        if not self.output_dir:
            return

        summary = {
            "experiment": self.experiment_config["experiment"],
            "timestamp": datetime.now().isoformat(),
            "output_directory": self.output_dir,
            "data_configs": [],
            "model_configs": [],
            "experiments_run": [],
        }

        # Add data config summaries
        for data_config_info in self.configs["data"]:
            summary["data_configs"].append(
                {
                    "name": data_config_info["name"],
                    "path": data_config_info["path"],
                    "features": list(
                        data_config_info["config"]["data"]["features"].keys()
                    ),
                }
            )

        # Add model config summaries
        for model_config_info in self.configs["models"]:
            model_config = model_config_info["config"]
            summary["model_configs"].append(
                {
                    "path": model_config_info["path"],
                    "model_type": model_config.get("model", {}).get("type", "unknown"),
                    "hypertuning": "hypertuning" in model_config,
                }
            )

        # Add experiment results summary
        for experiment_name, experiment_info in self.trained_pipelines.items():
            pipeline = experiment_info["pipeline"]
            best_score = None

            if hasattr(pipeline.named_steps["model"], "best_score_"):
                # Convert numpy scalar to regular Python float for human readability
                best_score = float(pipeline.named_steps["model"].best_score_)

            summary["experiments_run"].append(
                {
                    "name": experiment_name,
                    "data_config": experiment_info["data_config"],
                    "model_config": experiment_info["model_config"],
                    "model_name": experiment_info["model_name"],
                    "best_cv_score": best_score,
                }
            )

        # Save summary as YAML
        summary_file = os.path.join(self.output_dir, "experiment_summary.yaml")
        with open(summary_file, "w") as f:
            yaml.dump(summary, f, default_flow_style=False, indent=2)

        self.logger.info(f"Saved experiment summary to {summary_file}")

        # Save copies of original configuration files
        self.save_original_configs()

    def save_original_configs(self):
        """Save copies of all original configuration files used in the experiment."""
        if not self.output_dir:
            return

        configs_dir = os.path.join(self.output_dir, "configs")
        os.makedirs(configs_dir, exist_ok=True)

        # Save experiment config
        experiment_config_file = os.path.join(configs_dir, "experiment_config.yaml")
        with open(experiment_config_file, "w") as f:
            yaml.dump(self.experiment_config, f, default_flow_style=False, indent=2)
        self.logger.info(
            f"Saved original experiment config to {experiment_config_file}"
        )

        # Save data configs
        for data_config_info in tqdm(
            self.configs["data"], desc="Saving data configs", unit="config"
        ):
            config_name = data_config_info["name"]
            config_file = os.path.join(configs_dir, f"{config_name}.yaml")
            with open(config_file, "w") as f:
                yaml.dump(
                    data_config_info["config"], f, default_flow_style=False, indent=2
                )
            self.logger.info(
                f"Saved original data config '{config_name}' to {config_file}"
            )

        # Save model configs
        for model_config_info in tqdm(
            self.configs["models"], desc="Saving model configs", unit="config"
        ):
            config_path = model_config_info["path"]
            config_name = os.path.basename(config_path)
            config_file = os.path.join(configs_dir, config_name)
            with open(config_file, "w") as f:
                yaml.dump(
                    model_config_info["config"], f, default_flow_style=False, indent=2
                )
            self.logger.info(
                f"Saved original model config '{config_name}' to {config_file}"
            )

    def run_experiment(self):
        """Run the complete experiment pipeline."""
        self.logger.info("🚀 Starting Data Science Experiment...")
        self.logger.info(
            f"Experiment name: {self.experiment_config['experiment']['name']}"
        )
        self.logger.info(f"Number of data configs: {len(self.config_paths['data'])}")
        self.logger.info(f"Number of model configs: {len(self.model_config_paths)}")

        # Step 1: Load configurations
        self.logger.info("📂 Step 1: Loading configurations...")
        self.load_configs()
        self.logger.info("✅ Configurations loaded successfully")

        # Step 2: Setup data
        self.logger.info("📊 Step 2: Setting up data...")
        self.setup_data()
        self.logger.info("✅ Data setup completed")

        # Step 3: Setup preprocessing
        self.logger.info("🔧 Step 3: Setting up preprocessing pipelines...")
        self.setup_preprocessing()
        self.logger.info("✅ Preprocessing pipelines ready")

        # Step 4: Setup models
        self.logger.info("🤖 Step 4: Setting up models...")
        self.setup_models()
        self.logger.info("✅ Models setup completed")

        # Step 5: Train models
        self.logger.info("🏋️ Step 5: Training models...")
        self.train_models()
        self.logger.info("✅ Model training completed")

        # Step 6: Save results
        self.logger.info("💾 Step 6: Saving results...")
        self.save_models()
        self.save_hyperparameter_results()
        self.save_config_summary()
        self.logger.info("✅ Results saved successfully")

        self.logger.info("🎉 Experiment complete!")
        self.logger.info(f"All results saved to: {self.output_dir}")
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
    if controller.output_config and controller.output_dir:
        controller.save_models()
        controller.save_hyperparameter_results()
        controller.save_config_summary()
        print(f"\\n✓ All results saved to: {controller.output_dir}")
        print(f"  Models: {controller.models_dir}")
        print(f"  Results: {controller.results_dir}")
        print(f"  Logs: {controller.logs_dir}")
    else:
        print("\\n⚠ No output directory configured - results not saved")
