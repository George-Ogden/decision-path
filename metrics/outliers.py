import numpy as np
import torch

from typing import Dict

from .base import Metric, VariableLengthClassifierOutput

@Metric.register("outliers")
class Outliers(Metric):
    def __init__(self, threshold: float = 6.):
        self.threshold = threshold
        super().__init__()

    def update(self, batch: Dict[str, torch.Tensor], model_output: VariableLengthClassifierOutput):
        # L x [B, N, H]
        self.value += np.sum(
            [
                (activations > self.threshold).float().cpu().numpy().mean(axis=(-1, -2))
                for activations in model_output.layer_activations
            ],
            axis=1
        )
        super().update(batch, model_output)