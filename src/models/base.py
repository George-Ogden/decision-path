from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
from transformers.modeling_outputs import ModelOutput

from ..registry import Registry

@dataclass
class VariableLengthClassifierOutput(ModelOutput):
    layer_activations: Optional[List[torch.FloatTensor]] = None
    layer_predictions: Optional[torch.FloatTensor] = None

class VariableLengthModelForClassification(abc.ABC, nn.Module, Registry):
    registry: Dict[str, Type[VariableLengthModelForClassification]] = {}
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
        short_model_name = model_name.split("/")[-1]
        for key, model_class in cls.registry.items():
            if short_model_name.lower().startswith(key):
                return model_class._from_pretrained(model_name)
        raise ValueError(f"Model {model_name} not found in registry {cls.registry}.")

    @abc.abstractmethod
    def preprocess(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        ...