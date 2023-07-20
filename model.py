from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from torchvision.models import ResNet

@dataclass
class VariableLengthClassifierOutput(SequenceClassifierOutput):
    layer_predictions: Optional[torch.FloatTensor] = None

class VariableLengthModelForClassification(abc.ABC, nn.Module):
    @abc.abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> VariableLengthClassifierOutput:
        ...
    
    @abc.abstractproperty
    def layers(self) -> List[Tuple[int, int]]:
        """Returns a list of tuples of layer indices, where each tuple represents a layer group."""

    @abc.abstractstaticmethod
    def from_pretrained(model_name: str) -> VariableLengthModelForClassification:
        ...
    
class ReducedLengthModelForSequenceClassification(VariableLengthModelForClassification):
    @abc.abstractproperty
    def model(self) -> PreTrainedModel:
        ...

    @abc.abstractproperty
    def torso(self) -> nn.Module:
        ...
    
    @abc.abstractproperty
    def head(self) -> Optional[nn.Module]:
        ...
    
    def forward(self, *args: Any, **kwargs: Any) -> VariableLengthClassifierOutput:
        outputs: SequenceClassifierOutput = self.model(*args, **kwargs)
        predictions = None
        if self.head is not None:
            predictions = [self.head(hidden_state) for hidden_state in outputs.hidden_states]
        return VariableLengthClassifierOutput(
            **outputs,
            layer_predictions=predictions,
        )
    
    @property
    def layers(self) -> List[Tuple[int, int]]:
        return [
            (i, 0) for i in range(len(self.torso) + 1)
        ]