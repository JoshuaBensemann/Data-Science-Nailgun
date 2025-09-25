#!/usr/bin/env python3
"""
Script to run the reviews quantile 0.5 experiment.
"""

import sys
from pathlib import Path

# Add the project root to Python path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the controller
from ds_nailgun.nailgun.controller import run_experiment


def main():
    """Run the 0.5 quantile experiment."""
    print("🚀 Reviews Quantile 0.5 Experiment")
    print("=" * 50)

    # Path to the experiment configuration
    experiment_config_path = (
        "ds_nailgun/configs/examples/reviews_quantile_0.5_experiment_config_extra.yaml"
    )

    # Check if config file exists
    config_file = Path(project_root) / experiment_config_path
    if not config_file.exists():
        print(f"❌ Experiment config file not found: {config_file}")
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

        # Print summary
        print("\n📊 Experiment Summary:")
        print(f"   - Data configs processed: {len(controller.configs['data'])}")
        print(f"   - Model configs processed: {len(controller.configs['models'])}")
        print(f"   - Total experiments run: {len(controller.trained_pipelines)}")

        print("\n🔍 Trained pipelines:")
        for exp_name, exp_info in controller.trained_pipelines.items():
            model_name = exp_info["model_name"]
            data_config = exp_info["data_config"]
            print(f"   - {exp_name}: {model_name} on {data_config}")

    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
