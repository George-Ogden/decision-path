import numpy as np
import scipy.stats
import torch

from typing import Dict

from .base import Metric, VariableLengthClassifierOutput

@Metric.register("rms")
class RMS(Metric):
    def __init__(self):
        self.total = 0.
        self.count = 0

    def update(self, batch: Dict[str, torch.Tensor], model_output: VariableLengthClassifierOutput):
        # L x [B, N, H]
        self.total += np.sum(
            (model_output.layer_activations ** 2).cpu().numpy().mean(axis=(-1, -2)),
            axis=1
        )
        self.count += batch["batch_size"]

    def compute(self) -> float:
        return np.sqrt(self.total / self.count)