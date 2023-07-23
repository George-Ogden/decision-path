import numpy as np
import torch

from typing import Dict, List

from .base import Metric, VariableLengthClassifierOutput

@Metric.register("outliers")
class Outliers(Metric):
    def __init__(self, threshold: float = 6.):
        self.threshold = threshold
        super().__init__()
        self.value = None

    def update(self, batch: Dict[str, torch.Tensor], model_output: VariableLengthClassifierOutput):
        # L x [B, N, H]
        if self.value is None:
            self.value = [np.zeros(activation.shape[-1], dtype=bool) for activation in model_output.layer_activations]
        self.value = [
            self.value[i] | (activation > self.threshold).cpu().numpy().any(axis=(-3, -2))
            for i, activation in enumerate(model_output.layer_activations)
        ]
        super().update(batch, model_output)
    
    def compute(self) -> List[float]:
        return np.array([
            np.mean(value) for value in self.value
        ])