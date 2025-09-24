#!/usr/bin/env python3
"""
Demo script for post-experiment analysis using the Titanic dataset.

This script demonstrates how to use the PostExperimentController class
to analyze experiments and generate predictions.

Usage:
    # For installed package:
    python scripts/post_experiment_demo.py experiments/

    # For development (add project root to PYTHONPATH):
    PYTHONPATH=/path/to/project python scripts/post_experiment_demo.py experiments/
"""

import argparse
import sys
from pathlib import Path

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the controller class
from ds_nailgun.nailgun.post_experiment_controller import PostExperimentController


def main():
    """Main entry point for the demo script."""
    parser = argparse.ArgumentParser(
        description="Demo script for post-experiment analysis with Titanic data"
    )
    parser.add_argument(
        "folder_path",
        help="Path to folder containing experiments or single experiment directory",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompts for automated execution",
    )

    args = parser.parse_args()

    folder = Path(args.folder_path)
    if not folder.exists():
        print(f"❌ Folder does not exist: {folder}")
        sys.exit(1)

    try:
        # Create controller instance - it automatically discovers experiments
        controller = PostExperimentController(folder, force=args.force)

        # Run the analysis
        controller.run_analysis()

        print("\n✅ Demo completed successfully!")
        print("   The PostExperimentController class can be imported and used as:")
        print(
            "   from ds_nailgun.nailgun.post_experiment_controller import PostExperimentController"
        )
        print("   controller = PostExperimentController('path/to/experiments')")
        print("   controller.run_analysis()")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
