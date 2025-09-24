import yaml
from pathlib import Path
from collections import defaultdict
import pandas as pd
from sklearn.metrics import accuracy_score
from ds_nailgun.nailgun.post.model_load import load_model


class PostController:
    def __init__(self, folder_dir):
        self.folder_dir = Path(folder_dir)
        self.project_root = self.folder_dir.parent.parent
        self.summary_path = self.folder_dir / "experiment_summary.yaml"
        self._data_config_paths()
        self._models_paths()
        self._validate_models()

    def _data_config_paths(self):
        with open(self.summary_path, "r") as f:
            self.data = yaml.safe_load(f)
        self.data_configs = self.data.get("data_configs", [])
        for config in self.data_configs:
            print(config["path"])

    def _models_paths(self):
        models_dir = self.folder_dir / "models"
        if models_dir.exists():
            self.models = sorted(models_dir.iterdir())

    def _validate_models(self):
        experiments_run = self.data.get("experiments_run", [])
        config_to_models = defaultdict(list)
        for exp in experiments_run:
            config_name = exp["data_config"]
            model_name = exp["name"]
            config_to_models[config_name].append(model_name)
        for config in self.data_configs:
            name = config["name"]
            models = config_to_models.get(name, [])
            print(f"Data config {name}: {', '.join(models)}")

        # Test on validation data
        for config in self.data_configs:
            name = config["name"]
            config_path = self.project_root / config["path"]
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f)
            val_path = config_data["files"].get("validation_data")
            if not val_path:
                print(f"No validation data for {name}")
                continue
            val_df = pd.read_csv(self.project_root / val_path)
            target_col = config_data["data"]["target"]["column"]
            if target_col not in val_df.columns:
                print(
                    f"No '{target_col}' column in validation data for {name}, skipping."
                )
                continue
            features = [col for col in val_df.columns if col != target_col]
            true = val_df[target_col]
            models = config_to_models.get(name, [])
            for model_name in models:
                model_file = next(
                    (m for m in self.models if model_name in str(m)), None
                )
                if model_file:
                    model = load_model(str(model_file), silent=True)
                    pred = model.predict(val_df[features])
                    acc = accuracy_score(true, pred)
                    print(f"Model {model_name}: accuracy {acc:.4f}")
