#!/usr/bin/env python3
"""
Test script for the PreprocessingPipeline with configurable config path.
"""

import sys
import os
import argparse
import time

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from ds_nailgun.nailgun.preprocessing import create_preprocessing_pipeline


def test_preprocessing_pipeline(config_path):
    """Test the preprocessing pipeline with the given config."""
    try:
        print(f"Testing preprocessing pipeline with config: {config_path}")

        # Create preprocessing pipeline
        start_time = time.time()
        preprocessor = create_preprocessing_pipeline(config_path)
        pipeline = preprocessor.create_pipeline()
        setup_time = time.time() - start_time

        print(f"\n✅ Pipeline created successfully in {setup_time:.4f} seconds")
        print(f"Pipeline type: {type(pipeline).__name__}")

        # Analyze pipeline structure
        if hasattr(pipeline, "steps") and pipeline.steps:
            print(f"Pipeline steps: {len(pipeline.steps)}")
            for name, step in pipeline.steps:
                print(f"  - {name}: {type(step).__name__}")
                if (
                    hasattr(step, "transformers")
                    and hasattr(step, "__iter__")
                    and step.transformers
                ):
                    print(f"    Transformers: {len(step.transformers)}")
                    for trans_name, transformer, columns in step.transformers:
                        print(
                            f"      - {trans_name}: {type(transformer).__name__} on {len(columns)} columns"
                        )
                        if columns:
                            print(
                                f"        Columns: {columns[:5]}{'...' if len(columns) > 5 else ''}"
                            )

        elif hasattr(pipeline, "transformers") and pipeline.transformers:
            print(f"ColumnTransformer with {len(pipeline.transformers)} transformers:")
            for name, transformer, columns in pipeline.transformers:
                print(
                    f"  - {name}: {type(transformer).__name__} on {len(columns)} columns"
                )
                if columns:
                    print(
                        f"    Columns: {columns[:5]}{'...' if len(columns) > 5 else ''}"
                    )

        else:
            print("Pipeline: No-op transformer (passthrough)")

        # Show feature columns from config
        print("\nFeature columns from config:")
        for group_name, columns in preprocessor.feature_columns.items():
            print(f"  - {group_name}: {len(columns)} columns")
            if columns:
                print(f"    Columns: {columns[:5]}{'...' if len(columns) > 5 else ''}")

        # Test pipeline on dummy data (if possible)
        try:
            import pandas as pd
            import numpy as np

            # Create dummy data based on feature columns
            dummy_data = {}
            for group_name, columns in preprocessor.feature_columns.items():
                for col in columns:
                    if (
                        "age" in col.lower()
                        or "fare" in col.lower()
                        or "sold" in col.lower()
                        or "reviews" in col.lower()
                    ):
                        dummy_data[col] = np.random.normal(50, 20, 100)  # Numeric
                    elif (
                        "class" in col.lower()
                        or "sibsp" in col.lower()
                        or "parch" in col.lower()
                    ):
                        dummy_data[col] = np.random.randint(0, 5, 100)  # Integer
                    else:
                        dummy_data[col] = np.random.choice(
                            ["A", "B", "C"], 100
                        )  # Categorical

            if dummy_data:
                X_dummy = pd.DataFrame(dummy_data)
                print(
                    f"\nTesting pipeline on dummy data ({X_dummy.shape[0]} rows, {X_dummy.shape[1]} columns)"
                )

                # Fit and transform
                fit_start = time.time()
                pipeline.fit(X_dummy)
                X_transformed = pipeline.transform(X_dummy)
                fit_time = time.time() - fit_start

                print(
                    f"✅ Pipeline fit and transform successful in {fit_time:.4f} seconds"
                )
                print(f"Output shape: {X_transformed.shape}")

                if hasattr(X_transformed, "shape"):
                    print(
                        f"Input features: {X_dummy.shape[1]} → Output features: {X_transformed.shape[1]}"
                    )

        except Exception as e:
            print(f"\n⚠️  Could not test on dummy data: {str(e)}")
            print("This is normal if the config references external data files.")

        return True

    except Exception as e:
        print(f"❌ Error testing preprocessing pipeline: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test the PreprocessingPipeline with a config file."
    )
    parser.add_argument("config_path", help="Path to the YAML config file")

    args = parser.parse_args()

    success = test_preprocessing_pipeline(args.config_path)
    if success:
        print("\n✅ Preprocessing pipeline test passed!")
    else:
        print("\n❌ Preprocessing pipeline test failed!")
