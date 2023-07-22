from __future__ import annotations

import abc
from typing import Callable, Dict, List, Type

import torch

from ..models import VariableLengthClassifierOutput


class Metric(abc.ABC):
    registry = {}
    def __init__(self):
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
    
    @classmethod
    def register(cls, key: str) -> Callable[[Type[Metric]], Type[Metric]]:
        def decorator(model_class: Type[Metric]) -> Type[Metric]:
            cls.registry.append((key, model_class))
            return model_class
        return decorator
