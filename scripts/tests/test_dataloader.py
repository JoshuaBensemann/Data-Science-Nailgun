#!/usr/bin/env python3
"""
Test script for the DataLoader with configurable config path.
"""

import sys
import os
import argparse

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from ds_nailgun.nailgun.dataloader import DataLoader


def test_dataloader(config_path):
    """Test the dataloader with the given config."""
    try:
        print(f"Testing dataloader with config: {config_path}")

        # Initialize dataloader
        loader = DataLoader(config_path)

        # Load data
        data = loader.load_data()

        # Print results
        print("\nData loading successful!")
        print(f"Train data shape: {data['train'].shape}")
        print(f"Test data shape: {data['test'].shape}")

        if data["validation"] is not None:
            print(f"Validation data shape: {data['validation'].shape}")
        else:
            print("No validation data")

        print(f"\nTrain columns: {list(data['train'].columns)}")
        print(f"Test columns: {list(data['test'].columns)}")

        if data["validation"] is not None:
            print(f"Validation columns: {list(data['validation'].columns)}")

        # Show first few rows of train data
        print("\nFirst 5 rows of train data:")
        print(data["train"].head())

        return True

    except Exception as e:
        print(f"Error testing dataloader: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test the DataLoader with a config file."
    )
    parser.add_argument("config_path", help="Path to the YAML config file")

    args = parser.parse_args()

    success = test_dataloader(args.config_path)
    if success:
        print("\n✅ Dataloader test passed!")
    else:
        print("\n❌ Dataloader test failed!")
