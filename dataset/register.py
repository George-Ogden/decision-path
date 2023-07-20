from typing import Any

datasets = {}

def register(name: str, dataset: Any):
    datasets[name] = dataset