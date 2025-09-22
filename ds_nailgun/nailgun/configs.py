from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml

"""
Small YAML config loader for ds_nailgun.

Provides:
- read_yaml(path) -> dict: read and parse a YAML file (safe_load) and return a dict
- load_config(path=None, required=True) -> dict: look up default config files if path not provided
- parse_yaml_string(s) -> dict: parse YAML from a string

Requires PyYAML (yaml.safe_load).
"""


PathLike = Union[str, Path]


def read_yaml(path: PathLike) -> Dict[str, Any]:
    """
    Read and parse a YAML file and return its contents as a dict.

    Raises:
      FileNotFoundError: if the path does not exist
      TypeError: if the top-level YAML content is not a mapping
      yaml.YAMLError: if parsing fails
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"YAML file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise TypeError(
            f"YAML top-level content must be a mapping (dict), got {type(data).__name__}"
        )
    return data


def parse_yaml_string(s: str) -> Dict[str, Any]:
    """
    Parse YAML from a string and return a dict.
    """
    data = yaml.safe_load(s) or {}
    if not isinstance(data, dict):
        raise TypeError(
            f"YAML top-level content must be a mapping (dict), got {type(data).__name__}"
        )
    return data


def load_config(
    path: Optional[PathLike] = None, *, required: bool = True
) -> Dict[str, Any]:
    """
    Load a configuration dict.

    If path is provided, load that file. Otherwise search for common defaults in this order:
      - ./config.yaml
      - ./config.yml
      - ~/.nailgun.yaml

    If no file is found and required is True, FileNotFoundError is raised. If required is False,
    an empty dict is returned.
    """
    candidates = [
        Path.cwd() / "config.yaml",
        Path.cwd() / "config.yml",
        Path.home() / ".nailgun.yaml",
    ]

    if path:
        return read_yaml(path)

    for p in candidates:
        if p.exists():
            return read_yaml(p)

    if required:
        raise FileNotFoundError(
            "No config file found (tried config.yaml, config.yml, ~/.nailgun.yaml)"
        )
    return {}
