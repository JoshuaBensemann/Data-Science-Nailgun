#!/usr/bin/env python3
"""
Test script for experiment configurations with configurable config path.
"""

import sys
import os
import argparse
import time

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from ds_nailgun.nailgun.controller import ExperimentController


def test_experiment_config(config_path):
    """Test experiment configuration with the given config."""
    try:
        print(f"Testing experiment configuration: {config_path}")

        # Check if config file exists
        if not os.path.exists(config_path):
            print(f"❌ Config file does not exist: {config_path}")
            return False

        # Load and parse config
        import yaml

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        print("✅ Config file loaded successfully")
        print(f"Config structure: {list(config.keys())}")

        # Validate experiment section
        if "experiment" not in config:
            print("❌ Missing 'experiment' section")
            return False

        experiment_info = config["experiment"]
        print("\nExperiment details:")
        print(f"  - Name: {experiment_info.get('name', 'N/A')}")
        print(f"  - Description: {experiment_info.get('description', 'N/A')}")
        print(f"  - Random seed: {experiment_info.get('random_seed', 'N/A')}")

        # Validate data configs
        if "data" not in config or "config_paths" not in config["data"]:
            print("❌ Missing 'data.config_paths' section")
            return False

        data_configs = config["data"]["config_paths"]
        print(f"\nData configurations ({len(data_configs)}):")
        for i, data_path in enumerate(data_configs, 1):
            if os.path.exists(data_path):
                print(f"  ✅ {i}. {data_path}")
            else:
                print(f"  ❌ {i}. {data_path} (file not found)")

        # Validate model configs
        if "models" not in config or "config_paths" not in config["models"]:
            print("❌ Missing 'models.config_paths' section")
            return False

        model_configs = config["models"]["config_paths"]
        print(f"\nModel configurations ({len(model_configs)}):")
        for i, model_path in enumerate(model_configs, 1):
            if os.path.exists(model_path):
                print(f"  ✅ {i}. {model_path}")
            else:
                print(f"  ❌ {i}. {model_path} (file not found)")

        # Check logging config
        logging_config = config.get("logging", {})
        print("\nLogging configuration:")
        print(f"  - Level: {logging_config.get('level', 'INFO')}")
        print(f"  - File: {logging_config.get('file', 'N/A')}")

        # Check output config
        output_config = config.get("output", {})
        print("\nOutput configuration:")
        print(
            f"  - Base directory: {output_config.get('base_directory', 'experiments/')}"
        )
        print(f"  - Save format: {output_config.get('save_format', 'joblib')}")

        # Test controller initialization
        print("\nTesting controller initialization...")
        try:
            start_time = time.time()
            controller = ExperimentController(config_path)
            init_time = time.time() - start_time

            print(f"✅ Controller initialized successfully in {init_time:.4f} seconds")
            print(f"  - Data configs loaded: {len(controller.config_paths['data'])}")
            print(f"  - Model configs loaded: {len(controller.model_config_paths)}")
            print(f"  - Output directory: {controller.output_dir}")

        except Exception as e:
            print(f"❌ Controller initialization failed: {str(e)}")
            return False

        # Calculate total experiments
        total_experiments = len(data_configs) * len(model_configs)
        print("\nExperiment summary:")
        print(f"  - Total combinations: {total_experiments}")
        print(f"  - Data configs: {len(data_configs)}")
        print(f"  - Model configs: {len(model_configs)}")

        return True

    except Exception as e:
        print(f"❌ Error testing experiment config: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test experiment configuration with a config file."
    )
    parser.add_argument("config_path", help="Path to the YAML experiment config file")

    args = parser.parse_args()

    success = test_experiment_config(args.config_path)
    if success:
        print("\n✅ Experiment config test passed!")
    else:
        print("\n❌ Experiment config test failed!")
