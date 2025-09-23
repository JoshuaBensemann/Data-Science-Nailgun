#!/usr/bin/env python3
"""
Demo script for running a data science experiment using the Titanic dataset.

This script demonstrates how to use the ExperimentController class
to run a complete experiment with the Titanic survival prediction task.

Usage:
    # For installed package:
    python scripts/experiment_demo.py

    # For development (add project root to PYTHONPATH):
    PYTHONPATH=/path/to/project python scripts/experiment_demo.py
"""

import sys
from pathlib import Path

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the controller
from ds_nailgun.nailgun.controller import run_experiment


def main():
    """Main entry point for the experiment demo script."""
    print("🚀 Titanic Survival Prediction Experiment Demo")
    print("=" * 50)

    # Path to the Titanic experiment configuration
    experiment_config_path = (
        "ds_nailgun/configs/examples/titanic_experiment_config.yaml"
    )

    # Check if config file exists
    config_file = Path(project_root) / experiment_config_path
    if not config_file.exists():
        print(f"❌ Experiment config file not found: {config_file}")
        print("   Make sure you're running from the project root directory")
        sys.exit(1)

    print(f"📋 Using experiment config: {experiment_config_path}")

    try:
        # Run the experiment
        print("\n🏃 Starting experiment execution...\n")
        controller = run_experiment(experiment_config_path)

        print("\n✅ Experiment completed successfully!")
        print(f"📁 Results saved to: {controller.output_dir}")
        print(f"   Models: {controller.models_dir}")
        print(f"   Results: {controller.results_dir}")
        print(f"   Logs: {controller.logs_dir}")

        # Print summary of what was done
        print("\n📊 Experiment Summary:")
        print(f"   - Data configs processed: {len(controller.configs['data'])}")
        print(f"   - Model configs processed: {len(controller.configs['models'])}")
        print(f"   - Total experiments run: {len(controller.trained_pipelines)}")

        print("\n🔍 Trained pipelines:")
        for exp_name, exp_info in controller.trained_pipelines.items():
            model_name = exp_info["model_name"]
            data_config = exp_info["data_config"]
            print(f"   - {exp_name}: {model_name} on {data_config}")

        print("\n💡 Next steps:")
        print("   1. Check the experiment directory for detailed results")
        print("   2. Run post-experiment analysis:")
        print(f"      python scripts/post_experiment_demo.py {controller.output_dir}")
        print("   3. Review logs for detailed execution information")

        print("\n🎉 Demo completed successfully!")
        print("   The ExperimentController can be used as:")
        print("   from ds_nailgun.nailgun.controller import run_experiment")
        print("   controller = run_experiment('path/to/config.yaml')")

    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("   - Check that all required data files exist")
        print("   - Verify configuration file syntax")
        print("   - Ensure all dependencies are installed")
        print("   - Check logs for detailed error information")
        sys.exit(1)


if __name__ == "__main__":
    main()
