# Model config TEMPLATE

This consolidated template explains the general YAML schema used to preconfigure models for experiments. Use this as a guide for adding any model configuration.

- Quick contract
- Input: A YAML file describing a single model experiment (estimator, params, search).
- Output: A dictionary the experiment harness can use to instantiate an estimator and an optional hyperparameter search.
- Errors: Missing required keys, invalid parameter names, or parameter type mismatches should raise a validation error.

Top-level schema (recommended keys)
- name: string (friendly name for the configuration)
- model: string (short name for the model, e.g. `random_forest`, `xgboost_classifier`)
- estimator: string (fully-qualified import path or short class name, e.g. `sklearn.ensemble.RandomForestClassifier`)
- params: map/object — estimator init args
- search: optional object describing hyperparameter search
- metadata: optional map for author, date, notes

Detailed fields
- name (required)
  - Type: string
  - Example: "rfc-default"

- model (required)
  - Type: string
  - Purpose: logical id used by the harness to group or pick configs

- estimator (required)
  - Type: string
  - Should point to an importable class or a known alias recognized by the loader
  - Example: `sklearn.ensemble.RandomForestClassifier` or `xgboost.XGBClassifier`

- params (optional but recommended)
  - Type: mapping (string -> value)
  - Values: ints, floats, booleans, strings, lists, null (for None)
  - Example keys: `n_estimators`, `max_depth`, `learning_rate`, `random_state`



- search (optional)
  - Type: map
  - Keys:
    - method: string — `grid`, `random`, `bayes` (depending on harness support)
    - param_grid: mapping of parameter-name -> list-of-values
      - Parameter names usually need pipeline-style prefixes if using a pipeline (e.g., `estimator__n_estimators` or `pipeline__estimator__n_estimators`). Match the loader.
    - n_iter: integer (for random/bayes)
    - cv: integer cross-validation folds
    - scoring: string or list of strings
    - refit: string or bool to select metric for refitting
  - Example:
    method: grid
    param_grid:
      params__n_estimators: [100, 200, 500]
      params__max_depth: [null, 10, 20]
    cv: 5
    scoring: accuracy
    refit: accuracy

- metadata (optional)
  - author, created_on, notes

Examples

RandomForestClassifier (example)

```yaml
name: rfc-default
model: random_forest
estimator: sklearn.ensemble.RandomForestClassifier
params:
  n_estimators: 200
  criterion: gini
  max_depth: null
  random_state: 42
search:
  method: grid
  param_grid:
    params__n_estimators: [100, 200, 500]
    params__max_depth: [null, 10, 20]
  cv: 5
  scoring: accuracy
```

XGBoostClassifier (example)

```yaml
name: xgb-default
model: xgboost_classifier
estimator: xgboost.XGBClassifier
params:
  n_estimators: 300
  max_depth: 6
  learning_rate: 0.1
  subsample: 0.8
  colsample_bytree: 0.8
  random_state: 42
search:
  method: random
  param_grid:
    params__n_estimators: [100, 200, 300]
    params__learning_rate: [0.01, 0.05, 0.1]
  n_iter: 20
  cv: 5
  scoring: roc_auc
```

How to add new model configs
- Create a new YAML file in the `config_lib/models` folder with keys matching this template.
- Use `estimator` values that are importable in your runtime environment.
- If your project uses aliases for estimator types, add them consistently to the loader and templates.

Validation checklist (recommended automated checks)
- Required keys present: `name`, `model`, `estimator`.
- `params` keys are known/allowed by the estimator (optional strict mode).
- `search.param_grid` values are lists and non-empty.

Loader conventions and parameter prefixes
- Many experiment harnesses instantiate a `Pipeline` with steps followed by the estimator. In that case the hyperparameter grid must use names like `estimator__n_estimators` or `pipeline__estimator__n_estimators` depending on how the pipeline is named in code.
- If your loader accepts `params` as a sub-dictionary and wires them to the estimator automatically, prefer `params__<name>` keys in the search grid.

Appendix: common parameter examples by model type
- Tree-based (RandomForest, XGBoost): n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, subsample, colsample_bytree, learning_rate (xgboost)
- Linear models: penalty, C, solver
- Neural nets: hidden_layer_sizes, activation, learning_rate_init, epochs, batch_size
- Transformers: strategy (imputer), handle_unknown (encoder), with_mean/with_std (scaler)

Next steps I can take
- Create a small Python validator script that uses this schema to validate YAMLs in `config_lib/models` (I can add it under `ds_nailgun/nailgun/config_lib/scripts/validate_configs.py`).
- Update the README or add a short `HOWTO_ADD_MODEL.md` that walks contributors through adding configs.

---
If you'd like, I can now add the validator script and wire a small test that validates the existing `rfc.yaml` and `xgbc.yaml`. Which would you prefer?