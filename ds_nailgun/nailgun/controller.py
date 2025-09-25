import yaml
import logging
import joblib
import os
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    HalvingGridSearchCV,
    HalvingRandomSearchCV,
)
from sklearn.metrics import make_scorer, mean_pinball_loss
import pandas as pd
from .loaders.dataloader import DataLoader
from .loaders.preprocessing import create_preprocessing_pipeline
from .loaders.model_factory import create_model
from .optuna_search import OptunaSearchCV
from .consts import (
    HYPERTUNING_METHODS,
    SCORING_NAMES,
    SAVE_FORMATS,
    DEFAULT_BASE_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    LOGS_DIR,
    CONFIGS_DIR,
    EXPERIMENT_CONFIG_FILE,
    EXPERIMENT_SUMMARY_FILE,
    DEFAULT_CV_FOLDS,
    DEFAULT_GRID_SEARCH_VERBOSE,
    DEFAULT_PINBALL_ALPHA,
    DEFAULT_SCORING_NAME,
    DEFAULT_MODEL_TYPE,
    DEFAULT_LOG_LEVEL,
    DEFAULT_LOG_FORMAT,
    TIMESTAMP_FORMAT,
    DEFAULT_HALVING_RESOURCE,
    DEFAULT_HALVING_MAX_RESOURCES,
    DEFAULT_HALVING_N_CANDIDATES,
    DEFAULT_OPTUNA_N_TRIALS,
    DEFAULT_OPTUNA_TIMEOUT,
    DEFAULT_OPTUNA_N_JOBS,
    DEFAULT_OPTUNA_DIRECTION,
    DEFAULT_OPTUNA_STUDY_NAME,
)
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
        base_dir = self.output_config.get("base_directory", DEFAULT_BASE_DIR)
        os.makedirs(base_dir, exist_ok=True)

        # Create timestamped experiment directory
        experiment_name = (
            self.experiment_config["experiment"]["name"].replace(" ", "_").lower()
        )
        timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
        self.output_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")

        # Create subdirectories
        self.models_dir = os.path.join(self.output_dir, MODELS_DIR)
        self.results_dir = os.path.join(self.output_dir, RESULTS_DIR)
        self.logs_dir = os.path.join(self.output_dir, LOGS_DIR)

        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        self.output_base_dir = base_dir

    def setup_logging(self):
        """Setup logging based on experiment configuration."""
        logging_config = self.experiment_config.get("logging", {})
        level = getattr(logging, logging_config.get("level", DEFAULT_LOG_LEVEL).upper())
        format_str = logging_config.get("format", DEFAULT_LOG_FORMAT)

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

                columns_to_drop = [data_config["data"]["target"]["column"]]
                # Also drop ID column if it exists in the config
                if "id" in data_config["data"]:
                    id_column = data_config["data"]["id"]["column"]
                    if id_column in train_data.columns:
                        columns_to_drop.append(id_column)
                        self.logger.info(
                            f"Dropping ID column '{id_column}' from training data"
                        )

                X_train = train_data.drop(columns=columns_to_drop)
                y_train = train_data[data_config["data"]["target"]["column"]]

                for model_idx, (model, model_config_info) in enumerate(
                    zip(self.models, self.configs["models"])
                ):
                    experiment_count += 1
                    experiment_name = f"{data_config_name}_model_{model_idx + 1}"

                    try:
                        # Load the full model config to check for hypertuning
                        full_config = model_config_info["config"]

                        # Create base pipeline: preprocessing + model
                        base_pipeline = Pipeline(
                            [
                                (
                                    "preprocessing",
                                    self.preprocessing_pipeline[data_config_name],
                                ),
                                ("model", model),
                            ]
                        )

                        estimator = self._create_estimator_with_hypertuning(
                            base_pipeline,
                            full_config,
                            experiment_count,
                            data_config_name,
                            model,
                        )

                        # Use the estimator (which could be the base pipeline or a hypertuning wrapper)
                        pipeline = estimator

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

                        # Calculate validation score if validation data is available
                        validation_score, validation_metric = (
                            self._calculate_validation_score(
                                experiment_name,
                                pipeline,
                                data_config_name,
                                data_config,
                                columns_to_drop,
                                full_config,
                            )
                        )

                        # Store validation score
                        self.trained_pipelines[experiment_name]["validation_score"] = (
                            validation_score
                        )
                        self.trained_pipelines[experiment_name]["validation_metric"] = (
                            validation_metric
                        )

                        # Print best parameters if hypertuning was used
                        if isinstance(
                            pipeline,
                            (
                                GridSearchCV,
                                RandomizedSearchCV,
                                HalvingGridSearchCV,
                                HalvingRandomSearchCV,
                                OptunaSearchCV,
                            ),
                        ):
                            try:
                                # Safely access best_params_
                                best_params = getattr(pipeline, "best_params_", None)
                                if best_params is not None:
                                    self.logger.info(
                                        f"  Best parameters: {best_params}"
                                    )

                                # Safely access best_score_
                                best_score = getattr(pipeline, "best_score_", None)
                                if best_score is not None:
                                    self.logger.info(
                                        f"  Best cross-validation score: {best_score:.4f}"
                                    )
                            except Exception as e:
                                self.logger.warning(
                                    f"  Could not access best parameters/score: {str(e)}"
                                )

                        self.logger.info(
                            f"  Experiment {experiment_name} trained successfully"
                        )

                        # Save this model immediately after training
                        if self.output_dir:
                            self.save_single_model(experiment_name)
                            self.save_single_hyperparameter_results(experiment_name)
                            self.update_experiment_summary(experiment_name)

                    except Exception as e:
                        error_msg = (
                            f"❌ Failed to train experiment {experiment_name}: {str(e)}"
                        )
                        error_type_msg = f"   Error type: {type(e).__name__}"

                        # Print to console
                        print(error_msg)
                        print(error_type_msg)

                        # Log to file
                        self.logger.error(error_msg)
                        self.logger.error(error_type_msg)

                        # Continue to next model instead of stopping the entire training process

                    finally:
                        # Always update progress bar, even on failure
                        pbar.update(1)

        self.logger.info(f"Completed training {experiment_count} model experiments")
        return self.trained_pipelines

    def _calculate_validation_score(
        self,
        experiment_name,
        pipeline,
        data_config_name,
        data_config,
        columns_to_drop,
        full_config,
    ):
        """Calculate validation score for a trained pipeline."""
        validation_score = None
        validation_metric = None

        if self.data[data_config_name]["validation"] is not None:
            try:
                # Get validation data
                validation_data = self.data[data_config_name]["validation"].dropna(
                    subset=[data_config["data"]["target"]["column"]]
                )
                X_val = validation_data.drop(columns=columns_to_drop)
                y_val = validation_data[data_config["data"]["target"]["column"]]

                # Find the metric that was used for hypertuning
                # We'll use the same metric type when possible
                unique_target_values = len(set(y_val))
                is_classification = unique_target_values < 10

                # For hyperparameter search, we extract the scoring info
                # from the hypertuning_config directly to avoid attribute errors
                if "hypertuning" in full_config:
                    scoring_config = full_config["hypertuning"].get(
                        "scoring", {"name": DEFAULT_SCORING_NAME}
                    )
                    if isinstance(scoring_config, str):
                        validation_metric = scoring_config
                    elif isinstance(scoring_config, dict):
                        validation_metric = scoring_config.get(
                            "name", DEFAULT_SCORING_NAME
                        )
                else:
                    # Default metrics based on problem type
                    validation_metric = "accuracy" if is_classification else "r2"

                # Make predictions using the processed validation data
                y_pred = pipeline.predict(X_val)

                # Calculate score based on the metric
                if validation_metric == "accuracy" or (
                    validation_metric == DEFAULT_SCORING_NAME and is_classification
                ):
                    from sklearn.metrics import accuracy_score

                    validation_score = accuracy_score(y_val, y_pred)
                    self.logger.info(f"  Validation accuracy: {validation_score:.4f}")
                elif validation_metric == "r2" or validation_metric == "r2_score":
                    from sklearn.metrics import r2_score

                    validation_score = r2_score(y_val, y_pred)
                    self.logger.info(f"  Validation R²: {validation_score:.4f}")
                elif validation_metric == "neg_mean_squared_error":
                    from sklearn.metrics import mean_squared_error

                    mse = mean_squared_error(y_val, y_pred)
                    validation_score = -1.0 * mse  # Negate for consistent direction
                    self.logger.info(
                        f"  Validation MSE: {mse:.4f} (score: {validation_score:.4f})"
                    )
                elif (
                    validation_metric == "mean_pinball_loss"
                    or validation_metric in SCORING_NAMES
                ):
                    # For pinball loss, use the same alpha value as in hypertuning
                    if "hypertuning" in full_config:
                        scoring_config = full_config["hypertuning"].get(
                            "scoring", {"name": DEFAULT_SCORING_NAME}
                        )
                        alpha = (
                            scoring_config.get("alpha", DEFAULT_PINBALL_ALPHA)
                            if isinstance(scoring_config, dict)
                            else DEFAULT_PINBALL_ALPHA
                        )
                    else:
                        alpha = DEFAULT_PINBALL_ALPHA

                    pinball_loss = mean_pinball_loss(y_val, y_pred, alpha=alpha)
                    validation_score = (
                        -1.0 * pinball_loss
                    )  # Negate for consistent direction (higher is better)
                    self.logger.info(
                        f"  Validation pinball loss (alpha={alpha}): {pinball_loss:.4f} (score: {validation_score:.4f})"
                    )
                else:
                    # Default to standard metrics if we don't recognize the metric
                    if is_classification:
                        from sklearn.metrics import accuracy_score

                        validation_score = accuracy_score(y_val, y_pred)
                        validation_metric = "accuracy"
                        self.logger.info(
                            f"  Validation accuracy: {validation_score:.4f}"
                        )
                    else:
                        from sklearn.metrics import r2_score

                        validation_score = r2_score(y_val, y_pred)
                        validation_metric = "r2"
                        self.logger.info(f"  Validation R²: {validation_score:.4f}")

            except Exception as e:
                self.logger.warning(f"  Could not calculate validation score: {str(e)}")

        return validation_score, validation_metric

    def _create_estimator_with_hypertuning(
        self, base_pipeline, full_config, experiment_count, data_config_name, model
    ):
        """Create estimator with hypertuning configuration if specified."""
        estimator = base_pipeline
        if (
            "hypertuning" in full_config
            and full_config["hypertuning"]["method"] in HYPERTUNING_METHODS
        ):
            hypertuning_config = full_config["hypertuning"]

            # Handle special scoring metrics
            scoring_config = hypertuning_config.get(
                "scoring", {"name": DEFAULT_SCORING_NAME}
            )
            if isinstance(scoring_config, str):
                # Backward compatibility: if scoring is still a string
                scoring = scoring_config
            else:
                # New format: scoring is a dict with name and optional parameters
                scoring_name = scoring_config.get("name", DEFAULT_SCORING_NAME)
                if scoring_name in SCORING_NAMES:
                    alpha = scoring_config.get(
                        "alpha", DEFAULT_PINBALL_ALPHA
                    )  # Default to median if not specified
                    scoring = make_scorer(
                        mean_pinball_loss,
                        alpha=alpha,
                        greater_is_better=False,
                    )
                else:
                    scoring = scoring_name

            # Modify parameter grid to prefix with "model__" since parameters are nested in pipeline
            param_grid = hypertuning_config["parameters"]
            prefixed_param_grid = {}
            for key, values in param_grid.items():
                prefixed_param_grid[f"model__{key}"] = values

            method = hypertuning_config["method"]
            if method == "grid_search":
                estimator = GridSearchCV(
                    base_pipeline,
                    param_grid=prefixed_param_grid,
                    cv=hypertuning_config.get("cv", DEFAULT_CV_FOLDS),
                    scoring=scoring,
                    verbose=DEFAULT_GRID_SEARCH_VERBOSE,
                    n_jobs=hypertuning_config.get("n_jobs", -1),
                )
                self.logger.info(
                    f"Training experiment {experiment_count}: {data_config_name} + {type(model).__name__} with Grid Search"
                )
            elif method == "random_search":
                n_iter = hypertuning_config.get("n_iter", 10)  # Default 10 iterations
                estimator = RandomizedSearchCV(
                    base_pipeline,
                    param_distributions=prefixed_param_grid,
                    n_iter=n_iter,
                    cv=hypertuning_config.get("cv", DEFAULT_CV_FOLDS),
                    scoring=scoring,
                    verbose=DEFAULT_GRID_SEARCH_VERBOSE,
                    n_jobs=hypertuning_config.get("n_jobs", -1),
                    random_state=42,  # For reproducibility
                )
                self.logger.info(
                    f"Training experiment {experiment_count}: {data_config_name} + {type(model).__name__} with Random Search ({n_iter} iterations)"
                )
            elif method == "halving_grid_search":
                estimator = HalvingGridSearchCV(
                    base_pipeline,
                    param_grid=prefixed_param_grid,
                    cv=hypertuning_config.get("cv", DEFAULT_CV_FOLDS),
                    scoring=scoring,
                    verbose=DEFAULT_GRID_SEARCH_VERBOSE,
                    n_jobs=1,  # Halving doesn't support parallel well - force single job
                    random_state=42,  # For reproducibility
                    factor=hypertuning_config.get(
                        "factor", 3
                    ),  # Default halving factor for early stopping
                    resource=hypertuning_config.get(
                        "resource", DEFAULT_HALVING_RESOURCE
                    ),  # Resource to allocate
                    max_resources=hypertuning_config.get(
                        "max_resources", DEFAULT_HALVING_MAX_RESOURCES
                    ),  # Max resources parameter
                    min_resources=hypertuning_config.get(
                        "min_resources", "exhaust"
                    ),  # Min resources parameter - allow early stopping
                )
                self.logger.info(
                    f"Training experiment {experiment_count}: {data_config_name} + {type(model).__name__} with Halving Grid Search"
                )
            elif method == "halving_random_search":
                n_candidates = hypertuning_config.get(
                    "n_candidates", DEFAULT_HALVING_N_CANDIDATES
                )  # Default number of candidates
                estimator = HalvingRandomSearchCV(
                    base_pipeline,
                    param_distributions=prefixed_param_grid,
                    n_candidates=n_candidates,
                    cv=hypertuning_config.get("cv", DEFAULT_CV_FOLDS),
                    scoring=scoring,
                    verbose=DEFAULT_GRID_SEARCH_VERBOSE,
                    n_jobs=1,  # Halving doesn't support parallel well - force single job
                    random_state=42,  # For reproducibility
                    factor=hypertuning_config.get(
                        "factor", 3
                    ),  # Default halving factor for early stopping
                    resource=hypertuning_config.get(
                        "resource", DEFAULT_HALVING_RESOURCE
                    ),  # Resource to allocate
                    max_resources=hypertuning_config.get(
                        "max_resources", DEFAULT_HALVING_MAX_RESOURCES
                    ),  # Max resources parameter
                    min_resources=hypertuning_config.get(
                        "min_resources", "exhaust"
                    ),  # Min resources parameter - allow early stopping
                )
                self.logger.info(
                    f"Training experiment {experiment_count}: {data_config_name} + {type(model).__name__} with Halving Random Search ({n_candidates} candidates)"
                )
            elif method == "optuna":
                n_trials = hypertuning_config.get(
                    "n_trials", DEFAULT_OPTUNA_N_TRIALS
                )  # Default number of trials
                timeout = hypertuning_config.get(
                    "timeout", DEFAULT_OPTUNA_TIMEOUT
                )  # Default timeout
                direction = hypertuning_config.get(
                    "direction", DEFAULT_OPTUNA_DIRECTION
                )  # Default direction
                study_name = hypertuning_config.get(
                    "study_name", DEFAULT_OPTUNA_STUDY_NAME
                )  # Default study name
                n_jobs = hypertuning_config.get(
                    "n_jobs", DEFAULT_OPTUNA_N_JOBS
                )  # Default number of jobs

                # For Optuna, we need to transform the parameter distribution
                # from lists to dictionaries with parameter specifications
                # This allows Optuna to know what type each parameter should be
                estimator = OptunaSearchCV(
                    base_pipeline,
                    param_distributions=prefixed_param_grid,
                    n_trials=n_trials,
                    cv=hypertuning_config.get("cv", DEFAULT_CV_FOLDS),
                    scoring=scoring,
                    direction=direction,
                    timeout=timeout,
                    n_jobs=n_jobs,
                    study_name=study_name,
                    verbose=DEFAULT_GRID_SEARCH_VERBOSE,
                    random_state=42,  # For reproducibility
                )
                self.logger.info(
                    f"Training experiment {experiment_count}: {data_config_name} + {type(model).__name__} with Optuna ({n_trials} trials)"
                )
            self.logger.info(
                f"Training experiment {experiment_count}: {data_config_name} + {type(model).__name__}"
            )

        return estimator

    def save_single_model(self, experiment_name: str):
        """Save a single trained model pipeline."""
        if not self.output_dir or experiment_name not in self.trained_pipelines:
            return

        save_format = self.output_config.get("save_format", "joblib")
        experiment_info = self.trained_pipelines[experiment_name]
        pipeline = experiment_info["pipeline"]
        model_name = experiment_info["model_name"]

        # Get the actual model name (handle hypertuning)
        if hasattr(pipeline, "best_estimator_"):
            actual_model_name = type(
                pipeline.best_estimator_.named_steps["model"]
            ).__name__
        elif hasattr(pipeline.named_steps["model"], "best_estimator_"):
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

    def save_single_hyperparameter_results(self, experiment_name: str):
        """Save hyperparameter tuning results for a single experiment."""
        if not self.output_dir or experiment_name not in self.trained_pipelines:
            return

        experiment_info = self.trained_pipelines[experiment_name]
        pipeline = experiment_info["pipeline"]

        # Check if this pipeline used hypertuning
        # Pipeline could be GridSearchCV directly or contain a GridSearchCV in the model step
        if hasattr(pipeline, "cv_results_"):
            cv_results = pipeline.cv_results_
        elif hasattr(pipeline.named_steps["model"], "cv_results_"):
            cv_results = pipeline.named_steps["model"].cv_results_
        else:
            return

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

    def update_experiment_summary(self, experiment_name: str):
        """Update the experiment summary with a newly completed experiment."""
        if not self.output_dir or experiment_name not in self.trained_pipelines:
            return

        summary_file = os.path.join(self.output_dir, EXPERIMENT_SUMMARY_FILE)

        # Load existing summary or create new one
        if os.path.exists(summary_file):
            with open(summary_file, "r") as f:
                summary = yaml.safe_load(f)
        else:
            summary = {
                "experiment": self.experiment_config["experiment"],
                "timestamp": datetime.now().isoformat(),
                "output_directory": self.output_dir,
                "data_configs": [],
                "model_configs": [],
                "experiments_run": [],
            }

            # Add data config summaries (only once)
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

            # Add model config summaries (only once)
            for model_config_info in self.configs["models"]:
                model_config = model_config_info["config"]
                summary["model_configs"].append(
                    {
                        "path": model_config_info["path"],
                        "model_type": model_config.get("model", {}).get(
                            "type", DEFAULT_MODEL_TYPE
                        ),
                        "hypertuning": "hypertuning" in model_config,
                    }
                )

        # Add this experiment to the summary
        experiment_info = self.trained_pipelines[experiment_name]
        pipeline = experiment_info["pipeline"]
        best_score = None

        # Extract best_score safely
        try:
            if hasattr(pipeline, "best_score_"):
                best_score = float(pipeline.best_score_)
            elif hasattr(pipeline, "named_steps") and hasattr(
                pipeline.named_steps["model"], "best_score_"
            ):
                best_score = float(pipeline.named_steps["model"].best_score_)
        except Exception:
            self.logger.warning(f"Could not extract best_score_ for {experiment_name}")

        # Get validation score if available
        validation_score = experiment_info.get("validation_score", None)
        validation_metric = experiment_info.get("validation_metric", None)
        if validation_score is not None:
            validation_score = float(validation_score)

        # Check if this experiment is already in the summary
        existing_experiment = None
        for exp in summary["experiments_run"]:
            if exp["name"] == experiment_name:
                existing_experiment = exp
                break

        if existing_experiment:
            # Update existing entry
            update_data = {
                "data_config": experiment_info["data_config"],
                "model_config": experiment_info["model_config"],
                "model_name": experiment_info["model_name"],
                "best_cv_score": best_score,
            }
            # Add validation score if available
            if validation_score is not None:
                update_data["validation_score"] = validation_score
                if validation_metric is not None:
                    update_data["validation_metric"] = validation_metric

            existing_experiment.update(update_data)
        else:
            # Add new entry
            new_entry = {
                "name": experiment_name,
                "data_config": experiment_info["data_config"],
                "model_config": experiment_info["model_config"],
                "model_name": experiment_info["model_name"],
                "best_cv_score": best_score,
            }
            # Add validation score if available
            if validation_score is not None:
                new_entry["validation_score"] = validation_score
                if validation_metric is not None:
                    new_entry["validation_metric"] = validation_metric

            summary["experiments_run"].append(new_entry)

        # Save updated summary
        with open(summary_file, "w") as f:
            yaml.dump(summary, f, default_flow_style=False, indent=2)

        self.logger.info(f"Updated experiment summary with {experiment_name}")

    def save_models(self):
        """Save all remaining trained model pipelines (for backward compatibility)."""
        if not self.trained_pipelines or not self.output_dir:
            return

        for experiment_name in tqdm(
            self.trained_pipelines.keys(), desc="Saving remaining models", unit="model"
        ):
            self.save_single_model(experiment_name)

    def save_hyperparameter_results(self):
        """Save all remaining hyperparameter tuning results (for backward compatibility)."""
        if not self.trained_pipelines or not self.output_dir:
            return

        for experiment_name in tqdm(
            self.trained_pipelines.keys(),
            desc="Saving remaining hyperparameter results",
            unit="result",
        ):
            self.save_single_hyperparameter_results(experiment_name)

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
                    "model_type": model_config.get("model", {}).get(
                        "type", DEFAULT_MODEL_TYPE
                    ),
                    "hypertuning": "hypertuning" in model_config,
                }
            )

        # Add experiment results summary
        for experiment_name, experiment_info in self.trained_pipelines.items():
            pipeline = experiment_info["pipeline"]
            best_score = None

            # Extract best_score safely
            try:
                if hasattr(pipeline, "best_score_"):
                    best_score = float(pipeline.best_score_)
                elif hasattr(pipeline, "named_steps") and hasattr(
                    pipeline.named_steps["model"], "best_score_"
                ):
                    best_score = float(pipeline.named_steps["model"].best_score_)
            except Exception:
                self.logger.warning(
                    f"Could not extract best_score_ for {experiment_name}"
                )

            # Get validation score if available
            validation_score = experiment_info.get("validation_score", None)
            validation_metric = experiment_info.get("validation_metric", None)
            if validation_score is not None:
                validation_score = float(validation_score)

            # Create experiment summary entry
            experiment_entry = {
                "name": experiment_name,
                "data_config": experiment_info["data_config"],
                "model_config": experiment_info["model_config"],
                "model_name": experiment_info["model_name"],
                "best_cv_score": best_score,
            }

            # Add validation score if available
            if validation_score is not None:
                experiment_entry["validation_score"] = validation_score
                if validation_metric is not None:
                    experiment_entry["validation_metric"] = validation_metric

            summary["experiments_run"].append(experiment_entry)

        # Save summary as YAML
        summary_file = os.path.join(self.output_dir, EXPERIMENT_SUMMARY_FILE)
        with open(summary_file, "w") as f:
            yaml.dump(summary, f, default_flow_style=False, indent=2)

        self.logger.info(f"Saved experiment summary to {summary_file}")

        # Save copies of original configuration files
        self.save_original_configs()

    def save_original_configs(self):
        """Save copies of all original configuration files used in the experiment."""
        if not self.output_dir:
            return

        configs_dir = os.path.join(self.output_dir, CONFIGS_DIR)
        os.makedirs(configs_dir, exist_ok=True)

        # Save experiment config
        experiment_config_file = os.path.join(configs_dir, EXPERIMENT_CONFIG_FILE)
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
        try:
            self.logger.info("🚀 Starting Data Science Experiment...")
            self.logger.info(
                f"Experiment name: {self.experiment_config['experiment']['name']}"
            )
            self.logger.info(
                f"Number of data configs: {len(self.config_paths['data'])}"
            )
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

            # Step 5: Train models (includes incremental saving)
            self.logger.info("🏋️ Step 5: Training models...")
            self.train_models()
            self.logger.info("✅ Model training completed")

            # Step 6: Finalize results (configs and summary updates)
            self.logger.info("💾 Step 6: Finalizing results...")
            self.save_config_summary()
            self.logger.info("✅ Results finalized successfully")

            self.logger.info("🎉 Experiment complete!")
            self.logger.info(f"All results saved to: {self.output_dir}")
            return self.experiment_state

        except Exception as e:
            error_msg = f"❌ Experiment failed: {str(e)}"
            error_type_msg = f"   Error type: {type(e).__name__}"

            # Print to console
            print(error_msg)
            print(error_type_msg)

            # Log to file
            self.logger.error(error_msg)
            self.logger.error(error_type_msg)

            # Re-raise the exception
            raise

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
    try:
        controller = ExperimentController(experiment_config_path)
        controller.run_experiment()
        return controller
    except Exception as e:
        error_msg = f"❌ Experiment execution failed: {str(e)}"
        error_type_msg = f"   Error type: {type(e).__name__}"

        # Print to console
        print(error_msg)
        print(error_type_msg)

        # Re-raise the exception
        raise


# Example usage
if __name__ == "__main__":
    # Experiment configuration path
    experiment_config_path = (
        "ds_nailgun/configs/examples/titanic_experiment_config.yaml"
    )

    print(f"Running experiment from config: {experiment_config_path}")

    # Run experiment (models are saved incrementally during training)
    controller = run_experiment(experiment_config_path)

    # Check results
    if controller.output_config and controller.output_dir:
        print(f"\n✓ All results saved incrementally to: {controller.output_dir}")
        print(f"  Models: {controller.models_dir}")
        print(f"  Results: {controller.results_dir}")
        print(f"  Logs: {controller.logs_dir}")
        print(f"  Trained {len(controller.trained_pipelines)} model pipelines")
    else:
        print("\n⚠ No output directory configured - results not saved")
