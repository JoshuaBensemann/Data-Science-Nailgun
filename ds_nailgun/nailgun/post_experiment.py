#!/usr/bin/env python3
"""
Post-experiment analysis and prediction script.

This script loads trained models from an experiment directory,
evaluates them on validation data (if available), identifies the best
performing model, and generates predictions on test data.
"""

import argparse
import yaml
import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score
import sys


class PostExperimentAnalyzer:
    """Analyze completed experiments and generate predictions."""

    def __init__(self, experiment_dir: str):
        """
        Initialize analyzer with experiment directory.

        Args:
            experiment_dir: Path to the experiment directory
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_summary = None
        self.models = {}
        self.configs = {}
        self.data = {}

        # Validate experiment directory
        if not self.experiment_dir.exists():
            raise ValueError(f"Experiment directory does not exist: {experiment_dir}")

        required_files = ["experiment_summary.yaml", "configs", "models"]
        for file in required_files:
            if not (self.experiment_dir / file).exists():
                raise ValueError(f"Required file/directory missing: {file}")

        print(f"🔍 Analyzing experiment: {self.experiment_dir.name}")

    def load_experiment_summary(self):
        """Load the experiment summary YAML file."""
        summary_path = self.experiment_dir / "experiment_summary.yaml"
        with open(summary_path, "r") as f:
            self.experiment_summary = yaml.safe_load(f)

        print(
            f"📋 Loaded experiment summary with {len(self.experiment_summary['experiments_run'])} experiments"
        )

    def load_configs(self):
        """Load the original configuration files."""
        configs_dir = self.experiment_dir / "configs"

        # Load experiment config
        exp_config_path = configs_dir / "experiment_config.yaml"
        with open(exp_config_path, "r") as f:
            self.configs["experiment"] = yaml.safe_load(f)

        # Load data configs
        self.configs["data"] = {}
        for config_file in configs_dir.glob("data_config_*.yaml"):
            config_name = config_file.stem
            with open(config_file, "r") as f:
                self.configs["data"][config_name] = yaml.safe_load(f)

        # Load model configs
        self.configs["models"] = {}
        for config_file in configs_dir.glob("*_config.yaml"):
            if (
                not config_file.name.startswith("data_config")
                and config_file.name != "experiment_config.yaml"
            ):
                config_name = config_file.stem
                with open(config_file, "r") as f:
                    self.configs["models"][config_name] = yaml.safe_load(f)

        print(
            f"⚙️  Loaded {len(self.configs['data'])} data configs and {len(self.configs['models'])} model configs"
        )

    def load_models(self):
        """Load all trained model pipelines."""
        models_dir = self.experiment_dir / "models"

        for model_file in models_dir.glob("*.joblib"):
            model_name = model_file.stem
            try:
                model = joblib.load(model_file)
                self.models[model_name] = model
                print(f"🤖 Loaded model: {model_name}")
            except Exception as e:
                print(f"❌ Failed to load model {model_name}: {e}")

        print(f"📦 Total models loaded: {len(self.models)}")

    def load_data(self):
        """Load the original training and test data."""
        # Import here to avoid circular imports
        from .dataloader import DataLoader

        # Load data for each data config using the saved config files
        for data_config_name, data_config in self.configs["data"].items():
            try:
                # Use the saved config file path
                config_file_path = (
                    self.experiment_dir / "configs" / f"{data_config_name}.yaml"
                )

                # Create DataLoader instance with the config file path
                loader = DataLoader(str(config_file_path))

                # Load the data
                data = loader.load_data()
                self.data[data_config_name] = data

                print(f"📊 Loaded data for {data_config_name}:")
                for split_name, split_data in data.items():
                    if split_data is not None:
                        print(f"   {split_name}: {split_data.shape}")

            except Exception as e:
                print(f"❌ Failed to load data for {data_config_name}: {e}")

    def evaluate_models_on_validation(self):
        """Evaluate all models on validation data if available."""
        if (
            not self.experiment_summary
            or "experiments_run" not in self.experiment_summary
        ):
            print("⚠️  Experiment summary not loaded, skipping validation evaluation")
            return {}

        validation_results = {}

        for experiment in self.experiment_summary["experiments_run"]:
            exp_name = experiment["name"]
            data_config_name = experiment["data_config"]

            # Find the corresponding model
            model_key = None
            for model_name in self.models.keys():
                if exp_name in model_name:
                    model_key = model_name
                    break

            if model_key is None:
                print(f"⚠️  No model found for experiment {exp_name}")
                continue

            model = self.models[model_key]

            # Check if validation data exists
            if (
                data_config_name in self.data
                and "validation" in self.data[data_config_name]
            ):
                val_data = self.data[data_config_name]["validation"]
                if val_data is not None:
                    try:
                        # Make predictions
                        X_val = val_data.drop(
                            columns=[
                                self.configs["data"][data_config_name]["data"][
                                    "target_column"
                                ]
                            ]
                        )
                        y_val = val_data[
                            self.configs["data"][data_config_name]["data"][
                                "target_column"
                            ]
                        ]

                        y_pred = model.predict(X_val)
                        accuracy = accuracy_score(y_val, y_pred)

                        validation_results[exp_name] = {
                            "accuracy": accuracy,
                            "model_key": model_key,
                            "data_config": data_config_name,
                        }

                        print(
                            f"✅ Evaluated {exp_name} on validation: accuracy = {accuracy:.4f}"
                        )
                    except Exception as e:
                        print(f"❌ Failed to evaluate {exp_name} on validation: {e}")

        return validation_results

    def select_best_model(self, validation_results):
        """Select the best performing model."""
        if validation_results:
            # Use validation results if available
            best_exp = max(
                validation_results.keys(),
                key=lambda x: validation_results[x]["accuracy"],
            )
            best_result = validation_results[best_exp]
            print("\n🎯 Best model selected based on validation accuracy:")
            print(f"   Experiment: {best_exp}")
            print(f"   Validation Accuracy: {best_result['accuracy']:.4f}")
            print(f"   Model: {best_result['model_key']}")
            print(f"   Data config: {best_result['data_config']}")

            return best_exp, best_result["model_key"]

        else:
            # Use CV results from experiment summary
            if (
                not self.experiment_summary
                or "experiments_run" not in self.experiment_summary
            ):
                raise ValueError(
                    "Experiment summary not loaded or missing experiments_run data"
                )

            best_experiment = max(
                self.experiment_summary["experiments_run"],
                key=lambda x: x["best_cv_score"],
            )

            # Find the corresponding model
            exp_name = best_experiment["name"]
            model_key = None
            for model_name in self.models.keys():
                if exp_name in model_name:
                    model_key = model_name
                    break

            print("\n🎯 Best model selected based on CV scores (no validation data):")
            print(f"   Experiment: {exp_name}")
            print(f"   CV Score: {best_experiment['best_cv_score']:.4f}")
            print(f"   Model: {model_key}")
            print(f"   Data config: {best_experiment['data_config']}")

            return exp_name, model_key

    def generate_test_predictions(self, best_model_key, best_experiment):
        """Generate predictions on test data for the best model."""
        model = self.models[best_model_key]
        data_config_name = best_experiment["data_config"]

        if (
            data_config_name not in self.data
            or "test" not in self.data[data_config_name]
        ):
            print("❌ No test data available for predictions")
            return None

        test_data = self.data[data_config_name]["test"]
        if test_data is None:
            print("❌ Test data is None")
            return None

        try:
            # Prepare test data (assuming no target column in test data)
            X_test = test_data

            # Make predictions
            predictions = model.predict(X_test)

            # Create submission DataFrame
            submission = pd.DataFrame(
                {
                    "PassengerId": test_data.index
                    if hasattr(test_data, "index")
                    else range(len(predictions)),
                    "Survived": predictions.astype(int),
                }
            )

            # Save predictions
            output_file = self.experiment_dir / "test_predictions.csv"
            submission.to_csv(output_file, index=False)

            print(f"💾 Test predictions saved to: {output_file}")
            print(f"📊 Generated {len(predictions)} predictions")

            return submission

        except Exception as e:
            print(f"❌ Failed to generate test predictions: {e}")
            return None

    def display_model_details(self, model_key, experiment_info):
        """Display detailed information about a model."""
        print(f"\n🔍 Model Details: {model_key}")
        print(f"   Experiment: {experiment_info['name']}")
        print(f"   Data Config: {experiment_info['data_config']}")
        print(f"   Model Type: {experiment_info['model_name']}")
        print(f"   Best CV Score: {experiment_info['best_cv_score']:.4f}")
        # Try to show some model parameters
        try:
            model = self.models[model_key]
            if hasattr(model.named_steps["model"], "best_params_"):
                print(f"   Best Parameters: {model.named_steps['model'].best_params_}")
        except Exception:
            pass

    def run_analysis(self):
        """Run the complete post-experiment analysis."""
        try:
            # Load all necessary data
            self.load_experiment_summary()
            self.load_configs()
            self.load_models()
            self.load_data()

            # Evaluate models on validation data
            validation_results = self.evaluate_models_on_validation()

            # Select best model
            best_experiment_name, best_model_key = self.select_best_model(
                validation_results
            )

            # Get experiment info
            if (
                not self.experiment_summary
                or "experiments_run" not in self.experiment_summary
            ):
                raise ValueError(
                    "Experiment summary not loaded or missing experiments_run data"
                )

            best_experiment = next(
                exp
                for exp in self.experiment_summary["experiments_run"]
                if exp["name"] == best_experiment_name
            )

            # Display model details
            self.display_model_details(best_model_key, best_experiment)

            # Prompt user before generating predictions
            print("\n❓ Generate predictions on test data?")
            print(
                "   This will create a 'test_predictions.csv' file in the experiment directory."
            )
            response = input("   Proceed? (y/N): ").strip().lower()

            if response == "y":
                predictions = self.generate_test_predictions(
                    best_model_key, best_experiment
                )
                if predictions is not None:
                    print("✅ Test predictions generated successfully!")
                else:
                    print("❌ Failed to generate test predictions")
            else:
                print("⏭️  Skipping test prediction generation")

        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Post-experiment analysis and prediction generation"
    )
    parser.add_argument("experiment_dir", help="Path to the experiment directory")
    parser.add_argument(
        "--force", action="store_true", help="Skip confirmation prompts"
    )

    args = parser.parse_args()

    try:
        analyzer = PostExperimentAnalyzer(args.experiment_dir)
        analyzer.run_analysis()

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
