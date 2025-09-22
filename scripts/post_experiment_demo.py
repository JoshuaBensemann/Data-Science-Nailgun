#!/usr/bin/env python3
"""
Demo script for post-experiment analysis using the Titanic dataset.

This script demonstrates how to use the PostExperimentAnalyzer class
to analyze multiple experiments and generate predictions.

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

# Import the analyzer class
from ds_nailgun.nailgun.post_experiment import PostExperimentAnalyzer


def main():
    """Main entry point for the demo script."""
    parser = argparse.ArgumentParser(
        description="Demo script for post-experiment analysis with Titanic data"
    )
    parser.add_argument(
        "folder_dir",
        help="Path to folder containing experiments or single experiment directory",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompts for automated execution",
    )

    args = parser.parse_args()

    folder = Path(args.folder_dir)
    if not folder.exists():
        print(f"❌ Folder does not exist: {folder}")
        sys.exit(1)

    # Determine experiment directories
    experiment_dirs = []
    if (folder / "experiment_summary.yaml").exists():
        # Single experiment directory
        experiment_dirs = [str(folder)]
        print(f"🔍 Analyzing single experiment: {folder.name}")
    else:
        # Check for multiple experiments
        for subdir in folder.iterdir():
            if subdir.is_dir() and (subdir / "experiment_summary.yaml").exists():
                experiment_dirs.append(str(subdir))
        if not experiment_dirs:
            print(f"❌ No experiment directories found in {folder}")
            print("   Expected directories with 'experiment_summary.yaml' files")
            sys.exit(1)
        print(f"🔍 Analyzing {len(experiment_dirs)} experiments in {folder}")

    try:
        # Create analyzer instance
        analyzer = PostExperimentAnalyzer(experiment_dirs, force=args.force)

        # Run the analysis
        analyzer.run_analysis()

        print("\n✅ Demo completed successfully!")
        print("   The PostExperimentAnalyzer class can be imported and used as:")
        print(
            "   from ds_nailgun.nailgun.post_experiment import PostExperimentAnalyzer"
        )
        print("   analyzer = PostExperimentAnalyzer(['path/to/exp1', 'path/to/exp2'])")
        print("   analyzer.run_analysis()")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
