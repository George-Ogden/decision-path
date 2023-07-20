from __future__ import annotations

import abc
from typing import Callable, Dict, List, Type

import torch

from ..models import VariableLengthClassifierOutput


class Metric(abc.ABC):
    registry = {}
    @abc.abstractmethod
    def update(self, batch: Dict[str, torch.Tensor], model_output: VariableLengthClassifierOutput):
        """
        Args:
            batch (Dict[str, torch.Tensor]): batch from dataloader (including labels and batch_size)
            model_output (VariableLengthClassifierOutput): output of model
        """
        ...
    
    @abc.abstractmethod
    def compute(self) -> List[float]:
        """
        Returns:
            List[float]: value calculated for each layer
        """
        ...
    
    @classmethod
    def register(cls, key: str) -> Callable[[Type[Metric]], Type[Metric]]:
        def decorator(model_class: Type[Metric]) -> Type[Metric]:
            cls.registry.append((key, model_class))
            return model_class
        return decorator
