import numpy as np
import torch

from typing import Dict

from .base import Metric, VariableLengthClassifierOutput

@Metric.register("rms")
class RMS(Metric):
    def update(self, batch: Dict[str, torch.Tensor], model_output: VariableLengthClassifierOutput):
        # L x [B, N, H]
        self.value += np.array(
            [
                (activations ** 2).cpu().numpy().mean(axis=(-1, -2)).sum()
                for activations in model_output.layer_activations
            ],
        )
        super().update(batch, model_output)

    def compute(self) -> float:
        return np.sqrt(self.value / self.count)