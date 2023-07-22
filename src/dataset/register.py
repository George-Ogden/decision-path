from typing import Any

DATASETS = {}

def register(name: str, dataset: Any):
    DATASETS[name] = dataset