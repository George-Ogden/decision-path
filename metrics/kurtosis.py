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

@Metric.register("rotated_kurtosis")
class RotatedKurtosis(Kurtosis):
    def __init__(self):
        super().__init__()
        self.rotations = None

    def update(self, batch: Dict[str, torch.Tensor], model_output: VariableLengthClassifierOutput):
        # L x [B, N, H]
        if self.rotations is None:
            self.rotations = [
                torch.linalg.qr(torch.randn(activations.shape[-1], activations.shape[-1]))[0].to(activations.device)
                for activations in model_output.layer_activations
            ]
        for activation, rotation in zip(model_output.layer_activations, self.rotations):
            activation @= rotation
        super().update(batch, model_output)