"""
Post-experiment ensemble module for combining trained models.

This module provides functionality for creating ensemble models from previously trained models.
It supports various ensemble methods including voting, stacking, and weighted averaging.
"""

import os
import yaml
import joblib
import numpy as np
import logging
import pandas as pd
from typing import Dict, List, Optional, Any
from sklearn.ensemble import (
    VotingClassifier,
    VotingRegressor,
    StackingClassifier,
    StackingRegressor,
)
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from ..consts import SAVE_FORMATS

# Constants for ensemble configuration
ENSEMBLE_METHODS = ["voting", "stacking", "weighted"]
DEFAULT_ENSEMBLE_METHOD = "voting"
DEFAULT_ENSEMBLE_WEIGHTS = "uniform"  # Alternative: 'performance'
DEFAULT_ENSEMBLE_PASSTHROUGH = False  # For stacking: whether to pass original features


class EnsembleCreator:
    """
    Creates ensemble models from previously trained models.
    """

    def __init__(self, experiment_dir: str, logger: Optional[logging.Logger] = None):
        """
        Initialize ensemble creator with experiment directory.

        Args:
            experiment_dir: Path to the experiment directory containing models
            logger: Logger object for logging messages (optional)
        """
        self.experiment_dir = experiment_dir
        self.models_dir = os.path.join(experiment_dir, "models")
        self.results_dir = os.path.join(experiment_dir, "results")
        self.configs_dir = os.path.join(experiment_dir, "configs")

        # Setup logging
        if logger is None:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
        else:
            self.logger = logger

        # Load experiment summary
        self.summary_path = os.path.join(experiment_dir, "experiment_summary.yaml")
        if os.path.exists(self.summary_path):
            with open(self.summary_path, "r") as f:
                self.experiment_summary = yaml.safe_load(f)
        else:
            self.logger.error(f"Experiment summary not found at {self.summary_path}")
            self.experiment_summary = {}

        # Create directories for ensembles
        self.ensembles_dir = os.path.join(self.models_dir, "ensembles")
        os.makedirs(self.ensembles_dir, exist_ok=True)

        # Initialize containers
        self.trained_models = {}
        self.ensemble_configs = {}
        self.ensembles = {}

    def load_trained_models(
        self, model_paths: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Load trained models from the experiment directory.

        Args:
            model_paths: List of specific model paths to load. If None, load all models.

        Returns:
            Dictionary of loaded models with model names as keys.
        """
        if not os.path.exists(self.models_dir):
            self.logger.error(f"Models directory not found: {self.models_dir}")
            return {}

        # If specific models not provided, find all model files
        if model_paths is None:
            model_paths = []
            for format_ext in SAVE_FORMATS:
                model_paths.extend(
                    [
                        os.path.join(self.models_dir, f)
                        for f in os.listdir(self.models_dir)
                        if f.endswith(f".{format_ext}")
                        and os.path.isfile(os.path.join(self.models_dir, f))
                    ]
                )

        # Load each model
        for path in model_paths:
            try:
                model_name = os.path.splitext(os.path.basename(path))[0]
                model_format = os.path.splitext(path)[1][1:]  # Remove the dot

                if model_format == "joblib":
                    model = joblib.load(path)
                elif model_format == "pickle":
                    import pickle

                    with open(path, "rb") as f:
                        model = pickle.load(f)
                else:
                    self.logger.warning(f"Unsupported model format: {model_format}")
                    continue

                self.trained_models[model_name] = {"model": model, "path": path}

                self.logger.info(f"Loaded model: {model_name}")

            except Exception as e:
                self.logger.error(f"Failed to load model {path}: {str(e)}")

        return self.trained_models

    def load_ensemble_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load ensemble configuration from a YAML file.

        Args:
            config_path: Path to the ensemble configuration YAML file

        Returns:
            Dictionary containing ensemble configuration
        """
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            # Validate the config has required fields
            if "name" not in config:
                config["name"] = f"ensemble_{len(self.ensemble_configs) + 1}"

            if "method" not in config:
                config["method"] = DEFAULT_ENSEMBLE_METHOD

            if config["method"] not in ENSEMBLE_METHODS:
                self.logger.warning(
                    f"Invalid ensemble method: {config['method']}. Using default: {DEFAULT_ENSEMBLE_METHOD}"
                )
                config["method"] = DEFAULT_ENSEMBLE_METHOD

            # Store the config
            self.ensemble_configs[config["name"]] = config
            self.logger.info(f"Loaded ensemble config: {config['name']}")

            return config

        except Exception as e:
            self.logger.error(f"Failed to load ensemble config {config_path}: {str(e)}")
            return {}

    def _determine_task_type(self, model) -> bool:
        """
        Determine if a model is a classifier or regressor.

        Args:
            model: The model to check

        Returns:
            True if classifier, False if regressor
        """
        # Try multiple methods to determine model type
        if hasattr(model, "predict_proba"):
            return True

        # Check class name
        model_class = model.__class__.__name__.lower()
        if "classifier" in model_class:
            return True
        elif "regressor" in model_class:
            return False

        # If we can't determine, default to classifier
        self.logger.warning("Could not determine model type, assuming classifier")
        return True

    def _calculate_performance_weights(self, models_dict: Dict[str, Any]) -> np.ndarray:
        """
        Calculate weights based on model performance.

        Args:
            models_dict: Dictionary of models with performance metrics

        Returns:
            Array of normalized weights
        """
        weights = []

        for name, model_info in models_dict.items():
            # Try to find score in model info
            score = None

            # For scikit-learn search CV objects
            if hasattr(model_info["model"], "best_score_"):
                score = model_info["model"].best_score_

            # Look for score in experiment summary
            elif self.experiment_summary.get("models"):
                for model_summary in self.experiment_summary["models"]:
                    if model_summary["name"] == name and "score" in model_summary:
                        score = model_summary["score"]
                        break

            # Default weight if no score found
            if score is None:
                weights.append(1.0)
            else:
                # Transform negative scores (lower is better) to positive weights
                if score < 0:
                    score = 1.0 / (1.0 - score)
                weights.append(score)

        # Normalize weights
        weights = np.array(weights)
        if np.sum(weights) > 0:
            return weights / np.sum(weights)
        else:
            return np.ones(len(weights)) / len(weights)

    def create_ensemble(self, ensemble_config: Dict[str, Any]) -> BaseEstimator:
        """
        Create an ensemble model based on configuration.

        Args:
            ensemble_config: Dictionary containing ensemble configuration

        Returns:
            Scikit-learn compatible ensemble model
        """
        # Extract config parameters
        name = ensemble_config["name"]
        method = ensemble_config.get("method", DEFAULT_ENSEMBLE_METHOD)
        model_selection = ensemble_config.get("models", "all")
        weights_config = ensemble_config.get("weights", DEFAULT_ENSEMBLE_WEIGHTS)

        # Select models to include in ensemble
        if model_selection == "all":
            selected_models = self.trained_models
        elif isinstance(model_selection, list):
            selected_models = {
                name: self.trained_models[name]
                for name in model_selection
                if name in self.trained_models
            }
        else:
            self.logger.error(f"Invalid model selection: {model_selection}")
            return None

        if not selected_models:
            self.logger.error(f"No valid models found for ensemble {name}")
            return None

        # Check if models are compatible (all classifiers or all regressors)
        first_model = list(selected_models.values())[0]["model"]
        is_classifier = self._determine_task_type(first_model)

        # Create list of (name, estimator) tuples for ensemble
        estimators = []
        for model_name, model_info in selected_models.items():
            model = model_info["model"]

            # For scikit-learn search CV objects, get the best estimator
            if hasattr(model, "best_estimator_"):
                estimator = model.best_estimator_
            else:
                estimator = model

            estimators.append((model_name, estimator))

        # Calculate weights if needed
        if weights_config == "performance":
            weight_values = self._calculate_performance_weights(selected_models)
        elif weights_config == "uniform" or weights_config is None:
            weight_values = None
        elif isinstance(weights_config, list):
            weight_values = weights_config
        else:
            weight_values = None

        # Create the ensemble based on the specified method
        if method == "voting":
            if is_classifier:
                voting_type = ensemble_config.get(
                    "voting_type", "hard"
                )  # 'hard' or 'soft'
                ensemble_model = VotingClassifier(
                    estimators=estimators,
                    voting=voting_type,
                    weights=weight_values,
                    n_jobs=-1,
                )
            else:
                ensemble_model = VotingRegressor(
                    estimators=estimators, weights=weight_values, n_jobs=-1
                )

        elif method == "stacking":
            # Get final estimator if specified
            final_estimator = None
            if "final_estimator" in ensemble_config:
                # You'd need to implement loading the final estimator from config
                pass

            passthrough = ensemble_config.get(
                "passthrough", DEFAULT_ENSEMBLE_PASSTHROUGH
            )
            cv = ensemble_config.get("cv", 5)

            if is_classifier:
                ensemble_model = StackingClassifier(
                    estimators=estimators,
                    final_estimator=final_estimator,
                    cv=cv,
                    passthrough=passthrough,
                    n_jobs=-1,
                )
            else:
                ensemble_model = StackingRegressor(
                    estimators=estimators,
                    final_estimator=final_estimator,
                    cv=cv,
                    passthrough=passthrough,
                    n_jobs=-1,
                )

        elif method == "weighted":
            # Create a custom weighted ensemble using a specialized class
            class WeightedEnsemble(BaseEstimator):
                """Base class for weighted ensemble models."""

                def __init__(self, estimators, weights=None):
                    self.estimators = estimators
                    self.weights = (
                        weights
                        if weights is not None
                        else np.ones(len(estimators)) / len(estimators)
                    )

                def fit(self, X, y):
                    """Nothing to fit for pre-trained models."""
                    return self

                def predict(self, X):
                    """Make weighted prediction from all models."""
                    predictions = np.array(
                        [estimator.predict(X) for _, estimator in self.estimators]
                    )
                    return np.average(predictions, axis=0, weights=self.weights)

            if is_classifier:

                class WeightedEnsembleClassifier(WeightedEnsemble, ClassifierMixin):
                    """Weighted ensemble for classification tasks."""

                    def predict_proba(self, X):
                        """Make weighted probability predictions."""
                        try:
                            probas = np.array(
                                [
                                    estimator.predict_proba(X)
                                    for _, estimator in self.estimators
                                ]
                            )
                            return np.average(probas, axis=0, weights=self.weights)
                        except AttributeError:
                            self.logger.warning(
                                "Some estimators don't support predict_proba, using predict instead"
                            )
                            # Fall back to hard voting
                            return self.predict(X)

                ensemble_model = WeightedEnsembleClassifier(estimators, weight_values)
            else:

                class WeightedEnsembleRegressor(WeightedEnsemble, RegressorMixin):
                    """Weighted ensemble for regression tasks."""

                    pass

                ensemble_model = WeightedEnsembleRegressor(estimators, weight_values)
        else:
            self.logger.error(f"Unsupported ensemble method: {method}")
            return None

        # Store the ensemble
        self.ensembles[name] = {
            "model": ensemble_model,
            "config": ensemble_config,
            "model_names": list(selected_models.keys()),
        }

        return ensemble_model

    def train_ensemble(
        self, ensemble_name: str, X: pd.DataFrame, y: pd.Series
    ) -> BaseEstimator:
        """
        Train a created ensemble model on provided data.

        Args:
            ensemble_name: Name of the ensemble to train
            X: Features for training
            y: Target variable for training

        Returns:
            Trained ensemble model
        """
        if ensemble_name not in self.ensembles:
            self.logger.error(f"Ensemble {ensemble_name} not found")
            return None

        try:
            ensemble = self.ensembles[ensemble_name]["model"]

            # For voting/stacking ensembles that require fitting
            if hasattr(ensemble, "fit") and (
                isinstance(ensemble, VotingClassifier)
                or isinstance(ensemble, VotingRegressor)
                or isinstance(ensemble, StackingClassifier)
                or isinstance(ensemble, StackingRegressor)
            ):
                self.logger.info(f"Training ensemble {ensemble_name}...")
                ensemble.fit(X, y)
                self.logger.info(f"Ensemble {ensemble_name} trained successfully")

            return ensemble

        except Exception as e:
            self.logger.error(f"Failed to train ensemble {ensemble_name}: {str(e)}")
            return None

    def save_ensemble(self, ensemble_name: str, save_format: str = "joblib") -> str:
        """
        Save a trained ensemble model.

        Args:
            ensemble_name: Name of the ensemble to save
            save_format: Format to save the model ('joblib' or 'pickle')

        Returns:
            Path to the saved ensemble model
        """
        if ensemble_name not in self.ensembles:
            self.logger.error(f"Ensemble {ensemble_name} not found")
            return None

        if save_format not in SAVE_FORMATS:
            self.logger.warning(
                f"Unsupported save format: {save_format}. Using joblib."
            )
            save_format = "joblib"

        ensemble_info = self.ensembles[ensemble_name]
        ensemble_model = ensemble_info["model"]

        # Create filename and path
        filename = f"ensemble_{ensemble_name}.{save_format}"
        filepath = os.path.join(self.ensembles_dir, filename)

        try:
            # Save the model
            if save_format == "joblib":
                joblib.dump(ensemble_model, filepath)
            elif save_format == "pickle":
                import pickle

                with open(filepath, "wb") as f:
                    pickle.dump(ensemble_model, f)

            # Save ensemble configuration
            config_filepath = os.path.join(
                self.ensembles_dir, f"ensemble_{ensemble_name}_config.yaml"
            )
            with open(config_filepath, "w") as f:
                yaml.dump(
                    ensemble_info["config"], f, default_flow_style=False, indent=2
                )

            self.logger.info(f"Saved ensemble {ensemble_name} to {filepath}")

            # Update experiment summary
            self._update_experiment_summary(ensemble_name)

            return filepath

        except Exception as e:
            self.logger.error(f"Failed to save ensemble {ensemble_name}: {str(e)}")
            return None

    def _update_experiment_summary(self, ensemble_name: str) -> bool:
        """
        Update experiment summary with ensemble information.

        Args:
            ensemble_name: Name of the ensemble to add to summary

        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(self.summary_path) or ensemble_name not in self.ensembles:
            return False

        try:
            # Load current summary
            with open(self.summary_path, "r") as f:
                summary = yaml.safe_load(f)

            # Add ensembles section if it doesn't exist
            if "ensembles" not in summary:
                summary["ensembles"] = []

            # Check if this ensemble is already in the summary
            ensemble_exists = False
            for ens in summary["ensembles"]:
                if ens["name"] == ensemble_name:
                    ensemble_exists = True
                    break

            if not ensemble_exists:
                # Add this ensemble to the summary
                ensemble_info = self.ensembles[ensemble_name]

                summary["ensembles"].append(
                    {
                        "name": ensemble_name,
                        "method": ensemble_info["config"].get(
                            "method", DEFAULT_ENSEMBLE_METHOD
                        ),
                        "models": ensemble_info["model_names"],
                        "weights": ensemble_info["config"].get(
                            "weights", DEFAULT_ENSEMBLE_WEIGHTS
                        ),
                    }
                )

                # Save updated summary
                with open(self.summary_path, "w") as f:
                    yaml.dump(summary, f, default_flow_style=False, indent=2)

                self.logger.info(
                    f"Updated experiment summary with ensemble {ensemble_name}"
                )
                return True

            return False

        except Exception as e:
            self.logger.error(f"Failed to update experiment summary: {str(e)}")
            return False

    def evaluate_ensemble(
        self, ensemble_name: str, X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate an ensemble model on test data.

        Args:
            ensemble_name: Name of the ensemble to evaluate
            X: Features for evaluation
            y: Target variable for evaluation

        Returns:
            Dictionary of evaluation metrics
        """
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            mean_absolute_error,
            mean_squared_error,
            r2_score,
        )

        if ensemble_name not in self.ensembles:
            self.logger.error(f"Ensemble {ensemble_name} not found")
            return {}

        try:
            ensemble = self.ensembles[ensemble_name]["model"]
            is_classifier = self._determine_task_type(ensemble)

            # Make predictions
            y_pred = ensemble.predict(X)

            # Calculate metrics based on model type
            metrics = {}

            if is_classifier:
                # Classification metrics
                metrics["accuracy"] = accuracy_score(y, y_pred)

                # Check if binary or multiclass
                unique_classes = np.unique(y)
                if len(unique_classes) == 2:
                    # Binary classification
                    metrics["precision"] = precision_score(y, y_pred, average="binary")
                    metrics["recall"] = recall_score(y, y_pred, average="binary")
                    metrics["f1"] = f1_score(y, y_pred, average="binary")
                else:
                    # Multiclass classification
                    metrics["precision_macro"] = precision_score(
                        y, y_pred, average="macro"
                    )
                    metrics["recall_macro"] = recall_score(y, y_pred, average="macro")
                    metrics["f1_macro"] = f1_score(y, y_pred, average="macro")

                    metrics["precision_weighted"] = precision_score(
                        y, y_pred, average="weighted"
                    )
                    metrics["recall_weighted"] = recall_score(
                        y, y_pred, average="weighted"
                    )
                    metrics["f1_weighted"] = f1_score(y, y_pred, average="weighted")
            else:
                # Regression metrics
                metrics["mae"] = mean_absolute_error(y, y_pred)
                metrics["mse"] = mean_squared_error(y, y_pred)
                metrics["rmse"] = np.sqrt(metrics["mse"])
                metrics["r2"] = r2_score(y, y_pred)

            return metrics

        except Exception as e:
            self.logger.error(f"Failed to evaluate ensemble {ensemble_name}: {str(e)}")
            return {}
