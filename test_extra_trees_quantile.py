#!/usr/bin/env python3
"""
Test script for ExtraTreesQuantileRegressor with halving random search
"""

import yaml
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.metrics import mean_pinball_loss
from quantile_forest import ExtraTreesQuantileRegressor


def load_config(config_path):
    """Load YAML configuration file"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_sample_data(n_samples=1000, n_features=5):
    """Create sample regression data for testing"""
    np.random.seed(42)

    # Create features
    X = np.random.randn(n_samples, n_features)

    # Create target with some noise
    y = X[:, 0] * 2 + X[:, 1] * -1 + X[:, 2] * 0.5 + np.random.randn(n_samples) * 0.1

    return X, y


def test_extra_trees_quantile():
    """Test ExtraTreesQuantileRegressor with halving random search"""

    print("🔬 Testing ExtraTreesQuantileRegressor with Halving Random Search")
    print("=" * 60)

    # Load configuration
    config_path = (
        "ds_nailgun/configs/model_presets/extra_trees_quantile_0.5_config.yaml"
    )
    config = load_config(config_path)

    print(f"📋 Loaded config: {config_path}")
    print(
        f"🎯 Target quantile: {config['model']['parameters']['default_quantiles'][0]}"
    )
    print()

    # Create sample data
    print("📊 Creating sample data...")
    X, y = create_sample_data(n_samples=1000, n_features=5)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test set: {X_test.shape[0]} samples")
    print()

    # Create base model
    model_params = config["model"]["parameters"].copy()
    model_params.pop("default_quantiles")  # Remove from sklearn params
    quantiles = config["model"]["parameters"]["default_quantiles"]

    base_model = ExtraTreesQuantileRegressor(
        default_quantiles=quantiles, **model_params
    )

    print("🌲 Base model created:")
    print(f"   Type: {type(base_model).__name__}")
    print(f"   Quantiles: {quantiles}")
    print(f"   n_estimators: {model_params['n_estimators']}")
    print()

    # Setup hyperparameter tuning
    hypertuning_config = config["hypertuning"]

    # Use a simplified parameter grid for testing to avoid invalid combinations
    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", "log2"],
        "bootstrap": [True],
        "max_samples": [0.8, 1.0],
    }

    print("🔍 Setting up Halving Random Search:")
    print(f"   Method: {hypertuning_config['method']}")
    print(
        f"   Candidates: {hypertuning_config['n_candidates']} (from config, but using simplified grid)"
    )
    print(f"   CV folds: {hypertuning_config['cv']}")
    print(f"   Factor: {hypertuning_config['factor']}")
    print(f"   Scoring: pinball_loss (alpha={hypertuning_config['scoring']['alpha']})")
    print("   Using simplified parameter grid for testing")
    print()

    # Create scorer for pinball loss
    def pinball_loss_scorer(estimator, X, y):
        y_pred = estimator.predict(
            X, quantiles=[0.5]
        )  # Get median prediction - returns 1D for single quantile
        return -mean_pinball_loss(y, y_pred, alpha=0.5)  # Negative for maximization

    # Setup the search
    search = HalvingRandomSearchCV(
        base_model,
        param_distributions=param_grid,
        n_candidates=hypertuning_config["n_candidates"],
        cv=hypertuning_config["cv"],
        scoring=pinball_loss_scorer,
        factor=hypertuning_config["factor"],
        resource=hypertuning_config["resource"],
        random_state=42,
        verbose=1,
    )

    print("🚀 Starting hyperparameter search...")
    print()

    # Fit the search
    search.fit(X_train, y_train)

    print()
    print("✅ Search completed!")
    print(f"🏆 Best parameters: {search.best_params_}")
    print(".4f")
    print()

    # Evaluate on test set
    print("📈 Evaluating on test set...")
    best_model = search.best_estimator_
    y_pred_test = best_model.predict(
        X_test, quantiles=[0.5]
    )  # Returns 1D array for single quantile

    test_pinball_loss = mean_pinball_loss(y_test, y_pred_test, alpha=0.5)
    print(f"   Test pinball loss: {test_pinball_loss:.4f}")
    print()

    # Show some predictions
    print("🔮 Sample predictions (first 5 test samples):")
    for i in range(min(5, len(y_test))):
        print(".3f.3f.3f")

    print()
    print("🎉 Test completed successfully!")


if __name__ == "__main__":
    test_extra_trees_quantile()
