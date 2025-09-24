import joblib
import os


def load_model(model_path, silent=False):
    model = joblib.load(model_path)
    model_name = os.path.basename(model_path).replace(".joblib", "")
    model_name = model_name.replace("pipeline_", "")
    if not silent:
        print(f"Model: {model_name}")

        # Determine selected features
        if hasattr(model, "best_estimator_"):
            preprocessing = model.best_estimator_.named_steps.get("preprocessing")
            if preprocessing and hasattr(preprocessing, "transformer"):
                inner_pipeline = preprocessing.transformer
                if hasattr(inner_pipeline, "named_steps"):
                    selector = inner_pipeline.named_steps.get("feature_selection")
                    column_transformer = inner_pipeline.named_steps.get("preprocessing")
                    if (
                        selector
                        and hasattr(selector, "get_support")
                        and column_transformer
                        and hasattr(column_transformer, "get_feature_names_out")
                    ):
                        support = selector.get_support()
                        selected_indices = [i for i, s in enumerate(support) if s]
                        full_feature_names = column_transformer.get_feature_names_out()
                        selected_features = [
                            full_feature_names[i] for i in selected_indices
                        ]
                        print("Selected features:")
                        for feature in selected_features:
                            print(f"  {feature.split('__')[-1]}")
                        print()
                else:
                    print("Inner pipeline does not have named_steps attribute.")
            else:
                print("Preprocessing step not found in the pipeline.")
        else:
            print("Model does not have best_estimator_ attribute.")

    return model
