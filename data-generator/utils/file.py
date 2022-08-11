import os
import json
from typing import Dict


def get_filename(path: str):
    return os.path.splitext(os.path.basename(path))[0]


def extract_dataset_name(path: str) -> str:
    return os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(path))))


def save_data_dict(
    data_dict: Dict, output_dir: str, output_filename: str = "config.json"
):
    with open(os.path.join(output_dir, output_filename), "w") as f:
        serialized_config = {}
        for k, v in data_dict.items():
            serialized_config[k] = str(v)
        json.dump(serialized_config, f)
