from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, Type

import torch
import torch.nn as nn
from transformers.modeling_outputs import ModelOutput

@dataclass
class VariableLengthClassifierOutput(ModelOutput):
    layer_activations: Optional[List[torch.FloatTensor]] = None
    layer_predictions: Optional[torch.FloatTensor] = None

class VariableLengthModelForClassification(abc.ABC, nn.Module):
    registry: List[Tuple[str, Type[VariableLengthModelForClassification]]] = []
    @abc.abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> VariableLengthClassifierOutput:
        ...
    
    @abc.abstractproperty
    def layers(self) -> List[Tuple[int, int]]:
        """Returns a list of tuples of layer indices, where each tuple represents a layer group."""

    @abc.abstractclassmethod
    def _from_pretrained(cls, model_name: str) -> VariableLengthModelForClassification:
        ...
    
    @classmethod
    def from_pretrained(cls, model_name: str) -> VariableLengthModelForClassification:
        for key, model_class in cls.registry:
            if key in model_name.lower():
                return model_class._from_pretrained(model_name)
        raise ValueError(f"Model {model_name} not found in registry {cls.registry}.")

    @classmethod
    def register(cls, key: str) -> Callable[[Type[VariableLengthModelForClassification]], Type[VariableLengthModelForClassification]]:
        def decorator(model_class: Type[VariableLengthModelForClassification]) -> Type[VariableLengthModelForClassification]:
            cls.registry.append((key, model_class))
            return model_class
        return decorator