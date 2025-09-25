"""
Optuna search CV wrapper class for integration with scikit-learn.
This allows Optuna to be used as a drop-in replacement for scikit-learn's
grid search and randomized search classes.
"""

import numpy as np
import optuna
from sklearn.base import BaseEstimator, is_classifier
from sklearn.metrics import get_scorer
from sklearn.model_selection import check_cv
from sklearn.utils.validation import check_is_fitted


class OptunaSearchCV(BaseEstimator):
    """
    Hyperparameter optimization with Optuna.

    OptunaSearchCV implements a "fit" and a "score" method.
    It also implements "predict", "predict_proba", "decision_function",
    "transform" and "inverse_transform" if they are implemented in the
    estimator used.

    Parameters
    ----------
    estimator : estimator object
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    param_distributions : dict
        Dictionary with parameters names (string) as keys and distributions
        or lists of parameter settings to try. Distributions must provide a
        ``rvs`` method for sampling (such as those from scipy.stats.distributions).
        If a list is given, it is sampled uniformly.

        For Optuna, the dictionary values should be strings that indicate the
        parameter type and range, such as "float(0.01, 0.5)" for continuous params
        or "categorical([3, 5, 7, 9])" for discrete choices.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.

    scoring : str, callable, list/tuple or dict, default=None
        A single string or a callable to evaluate the predictions on the test set.

    direction : str, default="maximize"
        Direction of optimization. Set to "minimize" for metrics like MSE or "maximize"
        for metrics like accuracy or r2.

    n_trials : int, default=10
        Number of parameter settings that are sampled.

    timeout : int or None, default=None
        Maximum time (in seconds) for the search to run.

    n_jobs : int, default=None
        Number of jobs to run in parallel. -1 means using all processors.

    study_name : str, default="optuna_search"
        Name of the study for Optuna tracking.

    verbose : int, default=0
        Controls the verbosity: the higher, the more messages.

    random_state : int, default=None
        Random seed for Optuna sampler.
    """

    def __init__(
        self,
        estimator,
        param_distributions,
        *,
        cv=None,
        scoring=None,
        direction="maximize",
        n_trials=10,
        timeout=None,
        n_jobs=None,
        study_name="optuna_search",
        verbose=0,
        random_state=None,
    ):
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.cv = cv
        self.scoring = scoring
        self.direction = direction
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_jobs = n_jobs
        self.study_name = study_name
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X, y=None, **fit_params):
        """
        Run fit with all sets of parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
            Target relative to X for classification or regression.

        **fit_params : dict
            Additional parameters passed to the fit method of the estimator.

        Returns
        -------
        self : object
            Instance of fitted estimator.
        """
        cv = check_cv(self.cv, y, classifier=is_classifier(self.estimator))

        # Create sampler with specified random state
        sampler = optuna.samplers.TPESampler(seed=self.random_state)

        # Create study
        study = optuna.create_study(
            study_name=self.study_name,
            direction=self.direction,
            sampler=sampler,
        )

        # Get scorer
        if self.scoring is not None:
            scorer = get_scorer(self.scoring)
        else:
            scorer = get_scorer("accuracy" if is_classifier(self.estimator) else "r2")

        # Create the objective function
        def objective(trial):
            # Parse and sample parameters
            params = {}
            for param_name, param_specs in self.param_distributions.items():
                # Remove 'model__' prefix if present
                if param_name.startswith("model__"):
                    real_param_name = param_name[7:]  # Length of 'model__'
                else:
                    real_param_name = param_name

                # Parse param specs - this is a simple parser for common formats
                if isinstance(param_specs, list):
                    # If it's a list, treat as categorical choice
                    params[param_name] = trial.suggest_categorical(
                        real_param_name, param_specs
                    )
                elif isinstance(param_specs, dict):
                    # Handle dictionary of param distribution specifications
                    if "type" in param_specs:
                        if param_specs["type"] == "int":
                            params[param_name] = trial.suggest_int(
                                real_param_name,
                                param_specs["low"],
                                param_specs["high"],
                                step=param_specs.get("step", 1),
                                log=param_specs.get("log", False),
                            )
                        elif param_specs["type"] == "float":
                            if param_specs.get("log", False):
                                params[param_name] = trial.suggest_float(
                                    real_param_name,
                                    param_specs["low"],
                                    param_specs["high"],
                                    log=True,
                                )
                            else:
                                params[param_name] = trial.suggest_float(
                                    real_param_name,
                                    param_specs["low"],
                                    param_specs["high"],
                                    step=param_specs.get("step", None),
                                )
                        elif param_specs["type"] == "categorical":
                            params[param_name] = trial.suggest_categorical(
                                real_param_name, param_specs["choices"]
                            )
                        elif param_specs["type"] == "uniform":
                            # Uniform distribution (equivalent to float without log)
                            params[param_name] = trial.suggest_float(
                                real_param_name, param_specs["low"], param_specs["high"]
                            )
                        elif param_specs["type"] == "loguniform":
                            # Log-uniform distribution (equivalent to float with log=True)
                            params[param_name] = trial.suggest_float(
                                real_param_name,
                                param_specs["low"],
                                param_specs["high"],
                                log=True,
                            )
                        elif param_specs["type"] == "discrete_uniform":
                            # Discrete uniform distribution - float with step
                            params[param_name] = trial.suggest_float(
                                real_param_name,
                                param_specs["low"],
                                param_specs["high"],
                                step=param_specs["q"],
                            )
                        elif param_specs["type"] == "int_uniform":
                            # Integer uniform distribution
                            params[param_name] = trial.suggest_int(
                                real_param_name,
                                param_specs["low"],
                                param_specs["high"],
                                step=param_specs.get("step", 1),
                            )
                else:
                    # Backward compatibility - use the old string-based format
                    param_str = str(param_specs)
                    if param_str.startswith("int("):
                        # Format: int(low, high)
                        low, high = map(int, param_str[4:-1].split(","))
                        params[param_name] = trial.suggest_int(
                            real_param_name, low, high
                        )
                    elif param_str.startswith("float("):
                        # Format: float(low, high)
                        low, high = map(float, param_str[6:-1].split(","))
                        params[param_name] = trial.suggest_float(
                            real_param_name, low, high
                        )
                    elif param_str.startswith("categorical("):
                        # Format: categorical([val1, val2, ...])
                        values_str = param_str[11:-2]  # Remove 'categorical([' and '])'
                        values = eval(f"[{values_str}]")  # Convert string to list
                        params[param_name] = trial.suggest_categorical(
                            real_param_name, values
                        )
                    else:
                        # Default to categorical selection from the list
                        try:
                            values = eval(param_str)
                            if isinstance(values, list):
                                params[param_name] = trial.suggest_categorical(
                                    real_param_name, values
                                )
                        except Exception:
                            # If all else fails, treat as a literal string
                            params[param_name] = param_specs

            # Create a clone of the estimator with the sampled parameters
            estimator = self.estimator.set_params(**params)

            # Perform cross-validation
            scores = []
            for train_idx, test_idx in cv.split(X, y if y is not None else X):
                # Handle pandas DataFrames or numpy arrays
                if hasattr(X, "iloc"):
                    X_train = X.iloc[train_idx]
                    X_test = X.iloc[test_idx]
                else:
                    X_train = X[train_idx]
                    X_test = X[test_idx]

                if y is not None:
                    if hasattr(y, "iloc"):
                        y_train = y.iloc[train_idx]
                        y_test = y.iloc[test_idx]
                    else:
                        y_train = y[train_idx]
                        y_test = y[test_idx]
                else:
                    y_train = y_test = None

                estimator.fit(X_train, y_train, **fit_params)
                scores.append(scorer(estimator, X_test, y_test))

            # Return the mean score across folds as a float
            return float(np.mean(scores))

        # Optimize the objective function
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=1
            if self.n_jobs is None
            else int(self.n_jobs),  # Ensure n_jobs is an int
            show_progress_bar=(self.verbose > 0),
        )

        # Store the study
        self.study_ = study

        # Get best parameters and fit the best estimator
        self.best_params_ = study.best_params
        self.best_score_ = study.best_value
        self.best_trial_ = study.best_trial

        # Fit the best estimator on the entire dataset
        # Convert best_params to include model__ prefix
        best_params = {}
        for param_name, param_value in self.best_params_.items():
            if param_name in self.param_distributions:
                best_params[param_name] = param_value
            else:
                best_params[f"model__{param_name}"] = param_value

        self.best_estimator_ = self.estimator.set_params(**best_params)
        self.best_estimator_.fit(X, y, **fit_params)

        return self

    def score(self, X, y=None):
        """
        Return the score on the given data using the best estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
            Target relative to X for classification or regression.

        Returns
        -------
        score : float
            Score of the best estimator on X and y.
        """
        check_is_fitted(self, ["best_estimator_", "best_params_"])
        return self.best_estimator_.score(X, y)

    def predict(self, X):
        """
        Call predict on the estimator with the best found parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        y_pred : array-like
            Result of calling predict on the best estimator.
        """
        check_is_fitted(self, ["best_estimator_", "best_params_"])
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        """
        Call predict_proba on the estimator with the best found parameters.
        Only available if the estimator supports predict_proba.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        y_pred_proba : array-like
            Result of calling predict_proba on the best estimator.
        """
        check_is_fitted(self, ["best_estimator_", "best_params_"])
        return self.best_estimator_.predict_proba(X)

    def decision_function(self, X):
        """
        Call decision_function on the estimator with the best found parameters.
        Only available if the estimator supports decision_function.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        y_score : array-like
            Result of calling decision_function on the best estimator.
        """
        check_is_fitted(self, ["best_estimator_", "best_params_"])
        return self.best_estimator_.decision_function(X)

    def transform(self, X):
        """
        Call transform on the estimator with the best found parameters.
        Only available if the estimator supports transform.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        X_transformed : array-like
            Result of calling transform on the best estimator.
        """
        check_is_fitted(self, ["best_estimator_", "best_params_"])
        return self.best_estimator_.transform(X)

    def inverse_transform(self, X):
        """
        Call inverse_transform on the estimator with the best found parameters.
        Only available if the estimator supports inverse_transform.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        X_transformed : array-like
            Result of calling inverse_transform on the best estimator.
        """
        check_is_fitted(self, ["best_estimator_", "best_params_"])
        return self.best_estimator_.inverse_transform(X)
