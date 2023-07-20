import numpy as np
import scipy.stats
import torch

from typing import Dict

from .base import Metric, VariableLengthClassifierOutput

@Metric.register("kurtosis")
class Kurtosis(Metric):
    def __init__(self):
        self.kurtosis = 0.
        self.count = 0

    def update(self, batch: Dict[str, torch.Tensor], model_output: VariableLengthClassifierOutput):
        # L x [B, N, H]
        self.kurtosis += np.sum(
            [
                scipy.stats.kurtosis(activations.cpu().numpy(), axis=-1, fisher=False).mean(axis=-1)
                for activations in model_output.layer_activations
            ],
            axis=1
        )
        self.count += batch["batch_size"]

    def compute(self) -> float:
        return self.kurtosis / self.count