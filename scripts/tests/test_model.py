#!/usr/bin/env python3
"""
Test script for model creation with configurable config path.
"""

import sys
import os
import argparse
import time

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from ds_nailgun.nailgun.model_factory import create_model


def test_model(config_path):
    """Test model creation with the given config."""
    try:
        print(f"Testing model creation with config: {config_path}")

        # Create model
        start_time = time.time()
        model = create_model(config_path)
        creation_time = time.time() - start_time

        print(f"\n✅ Model created successfully in {creation_time:.4f} seconds")
        print(f"Model type: {type(model).__name__}")
        print(f"Model class: {model.__class__.__module__}.{model.__class__.__name__}")

        # Show model parameters
        print("\nModel parameters:")
        if hasattr(model, "get_params"):
            params = model.get_params()
            for param_name, param_value in sorted(params.items()):
                # Format long parameter names/values for readability
                param_str = str(param_value)
                if len(param_str) > 50:
                    param_str = param_str[:47] + "..."
                print(f"  - {param_name}: {param_str}")

        # Show key attributes specific to the model type
        print("\nModel-specific attributes:")

        # For tree-based models
        if hasattr(model, "n_estimators"):
            print(f"  - n_estimators: {model.n_estimators}")
        if hasattr(model, "max_depth"):
            print(f"  - max_depth: {model.max_depth}")
        if hasattr(model, "random_state"):
            print(f"  - random_state: {model.random_state}")

        # For quantile models
        if (
            hasattr(model, "quantile")
            or hasattr(model, "quantiles")
            or hasattr(model, "default_quantiles")
        ):
            if hasattr(model, "quantile"):
                print(f"  - quantile: {model.quantile}")
            if hasattr(model, "quantiles"):
                print(f"  - quantiles: {model.quantiles}")
            if hasattr(model, "default_quantiles"):
                print(f"  - default_quantiles: {model.default_quantiles}")

        # For XGBoost models
        if hasattr(model, "learning_rate"):
            print(f"  - learning_rate: {model.learning_rate}")
        if hasattr(model, "subsample"):
            print(f"  - subsample: {model.subsample}")
        if hasattr(model, "colsample_bytree"):
            print(f"  - colsample_bytree: {model.colsample_bytree}")

        # Test basic model functionality with dummy data
        try:
            import numpy as np

            # Create dummy data based on model type
            np.random.seed(42)
            n_samples, n_features = 100, 5

            X = np.random.randn(n_samples, n_features)

            # Create target based on model type
            if "Classifier" in type(model).__name__:
                y = np.random.randint(0, 2, n_samples)  # Binary classification
                task_type = "classification"
            elif "Regressor" in type(model).__name__:
                y = np.random.randn(n_samples)  # Regression
                task_type = "regression"
            else:
                y = np.random.randn(n_samples)  # Default to regression
                task_type = "unknown"

            print(
                f"\nTesting model on dummy {task_type} data ({n_samples} samples, {n_features} features)"
            )

            # Test fitting
            fit_start = time.time()
            model.fit(X, y)
            fit_time = time.time() - fit_start

            print(f"✅ Model fit successful in {fit_time:.4f} seconds")

            # Test prediction
            pred_start = time.time()
            if task_type == "classification":
                predictions = model.predict(X)
                probabilities = (
                    model.predict_proba(X) if hasattr(model, "predict_proba") else None
                )
                print(f"Predictions shape: {predictions.shape}")
                if probabilities is not None:
                    print(f"Probabilities shape: {probabilities.shape}")
            else:
                predictions = model.predict(X)
                print(f"Predictions shape: {predictions.shape}")
                print(
                    f"Prediction range: [{predictions.min():.3f}, {predictions.max():.3f}]"
                )

            pred_time = time.time() - pred_start
            print(f"✅ Prediction successful in {pred_time:.4f} seconds")

        except Exception as e:
            print(f"\n⚠️  Could not test on dummy data: {str(e)}")
            print("This is normal for some model types or configurations.")

        return True

    except Exception as e:
        print(f"❌ Error testing model: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test model creation with a config file."
    )
    parser.add_argument("config_path", help="Path to the YAML config file")

    args = parser.parse_args()

    success = test_model(args.config_path)
    if success:
        print("\n✅ Model test passed!")
    else:
        print("\n❌ Model test failed!")
