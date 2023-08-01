from __future__ import annotations

from typing import Dict, List
import abc

import torch

from ..models import VariableLengthClassifierOutput
from ..registry import Registry

class Metric(abc.ABC, Registry):
    registry: Dict[str, Metric] = {}
    def __init__(self):
        # initialize count and value
        self.count = 0.
        self.value = 0.
        
    @abc.abstractmethod
    def update(self, batch: Dict[str, torch.Tensor], model_output: VariableLengthClassifierOutput):
        """
        Args:
            batch (Dict[str, torch.Tensor]): batch from dataloader (including labels and batch_size)
            model_output (VariableLengthClassifierOutput): output of model
        """
        self.count += batch["batch_size"]

    def compute(self) -> List[float]:
        """
        Returns:
            List[float]: value calculated for each layer
        """
        return self.value / self.count
