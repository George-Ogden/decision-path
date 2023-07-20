import abc
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple
import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput


@dataclass
class VariableLengthSequenceClassifierOutput(SequenceClassifierOutput):
    layer_predictions: Optional[torch.FloatTensor] = None

class VariableLengthModelForSequenceClassification(abc.ABC, nn.Module):
    @abc.abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> VariableLengthSequenceClassifierOutput:
        ...
    
    @abc.abstractproperty
    def layers(self) -> List[Tuple[int, int]]:
        """Returns a list of tuples of layer indices, where each tuple represents a layer group."""