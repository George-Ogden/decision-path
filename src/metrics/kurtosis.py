import numpy as np
import scipy.stats
import torch

from typing import Dict

from .base import Metric, VariableLengthClassifierOutput

@Metric.register("kurtosis")
class Kurtosis(Metric):
    """Mean kurtosis of activations."""
    def update(self, batch: Dict[str, torch.Tensor], model_output: VariableLengthClassifierOutput):
        # L x [B, N, H]
        self.value += np.array([
            scipy.stats.kurtosis(activations.cpu().numpy(), axis=-1, fisher=False).mean(axis=-1).sum()
            for activations in model_output.layer_activations
        ])
        super().update(batch, model_output)

@Metric.register("rotated-kurtosis")
class RotatedKurtosis(Kurtosis):
    """Mean kurtosis of activations after random rotation."""
    def __init__(self):
        super().__init__()
        self.rotations = None

    def update(self, batch: Dict[str, torch.Tensor], model_output: VariableLengthClassifierOutput):
        # L x [B, N, H]
        if self.rotations is None:
            # generate a random rotation by QR decomposing a random matrix
            self.rotations = [
                torch.linalg.qr(torch.randn(activations.shape[-1], activations.shape[-1]))[0].to(activations.device)
                for activations in model_output.layer_activations
            ]
        # rotate activations
        model_output.layer_activations = [
            activation @ rotation
            for activation, rotation in zip(model_output.layer_activations, self.rotations)
        ]
        super().update(batch, model_output)