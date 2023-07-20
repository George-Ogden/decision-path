import numpy as np
import torch

from typing import Dict

from .base import Metric, VariableLengthClassifierOutput

@Metric.register("outliers")
class Outliers(Metric):
    def __init__(self, threshold: float = 6.):
        self.outliers = 0
        self.count = 0
        self.threshold = threshold

    def update(self, batch: Dict[str, torch.Tensor], model_output: VariableLengthClassifierOutput):
        # L x [B, N, H]
        self.outliers += np.sum(
            [
                (activations > self.threshold).float().cpu().numpy().mean(axis=(-1, -2))
                for activations in model_output.layer_activations
            ],
            axis=1
        )
        self.count += batch["batch_size"]

    def compute(self) -> float:
        return self.outliers / self.count