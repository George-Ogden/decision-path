import numpy as np
import torch

from typing import Dict, List

from .base import Metric, VariableLengthClassifierOutput

@Metric.register("outliers")
class Outliers(Metric):
    """Proportion of channels with activations above a threshold."""
    def __init__(self, threshold: float = 6.):
        # threshold for outlier detection
        self.threshold = threshold
        super().__init__()
        # start with None rather than 0
        self.value = None

    def update(self, batch: Dict[str, torch.Tensor], model_output: VariableLengthClassifierOutput):
        # L x [B, N, H]
        if self.value is None:
            # store values for each channel of each layer
            self.value = [np.zeros(activation.shape[-1], dtype=bool) for activation in model_output.layer_activations]
        # L x [H]
        self.value = [
            # register all activations that exceeded the threshold in their respective channels
            self.value[i] | (activation > self.threshold).cpu().numpy().any(axis=(-3, -2))
            for i, activation in enumerate(model_output.layer_activations)
        ]
        super().update(batch, model_output)
    
    def compute(self) -> List[float]:
        # compute the mean of each layer
        return np.array([
            np.mean(value) for value in self.value
        ])