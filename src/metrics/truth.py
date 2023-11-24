import numpy as np
import torch

from typing import Dict, List

from .base import Metric, VariableLengthClassifierOutput

@Metric.register("truth")
class Truth(Metric):
    """Truth of boolq statements."""
    def __init__(self):
        super().__init__()
        self.values = []

    def update(self, batch: Dict[str, torch.Tensor], model_output: VariableLengthClassifierOutput):
        super().update(batch, model_output)
        # [B]
        self.values.append(
            batch["labels"].detach().cpu().numpy()
        )
        # NB x [B]
    
    def compute(self) -> List[List[float]]:
        return np.concatenate(self.values, axis=0)
