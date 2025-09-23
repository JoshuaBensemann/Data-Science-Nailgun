import yaml
import importlib
import logging


def create_model(config_path):
    """
    Create a machine learning model from a YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        model: The instantiated model object.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Creating model from config: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_config = config["model"]
    library = model_config["library"]
    model_type = model_config["type"]
    parameters = model_config["parameters"]

    logger.debug(f"Model library: {library}, type: {model_type}")
    logger.debug(f"Model parameters: {parameters}")

    # Import the library module
    module = importlib.import_module(library)
    logger.debug(f"Imported module: {module}")

    # Get the model class
    model_class = getattr(module, model_type)
    logger.debug(f"Model class: {model_class}")

    # Instantiate the model with parameters
    model = model_class(**parameters)
    logger.info(f"Model created: {type(model).__name__}")

    return model


if __name__ == "__main__":
    # Test the model factory with both presets
    rf_config_path = (
        "ds_nailgun/configs/model_presets/random_forest_classifier_config.yaml"
    )
    xgb_config_path = "ds_nailgun/configs/model_presets/xgboost_classifier_config.yaml"

    print("Testing Random Forest Model:")
    rf_model = create_model(rf_config_path)
    print(f"Type: {type(rf_model)}")
    print(f"Model: {rf_model}")
    print()

    print("Testing XGBoost Model:")
    xgb_model = create_model(xgb_config_path)
    print(f"Type: {type(xgb_model)}")
    print(f"Model: {xgb_model}")
