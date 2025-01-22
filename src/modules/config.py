# Standard Imports
from typing import Any
from tomllib import load


def get_config(toml_file_path: str) -> dict[str, Any]:
    with open(toml_file_path, "rb") as f:
        config = load(f)
    return config
