from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput, SequenceClassifierOutput
from torchvision.models import ResNet

@dataclass
class VariableLengthClassifierOutput(ModelOutput):
    layer_activations: Optional[List[torch.FloatTensor]] = None
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
            layer_activations=outputs.hidden_states,
            layer_predictions=predictions,
        )
    
    @property
    def layers(self) -> List[Tuple[int, int]]:
        return [
            (i, 0) for i in range(len(self.torso) + 1)
        ]

class ReducedLengthModelForImageClassification(VariableLengthModelForClassification):
    def __init__(self, model: ResNet):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
        )
        self.torso: List[nn.Module] = [
            getattr(model, name)
            for name in model._modules.keys()
            if name.startswith("layer")
        ]
        self.head = nn.Sequential(
            model.avgpool,
            nn.Flatten(1),
            model.fc,
        )

    def forward(self, x: torch.Tensor) -> VariableLengthClassifierOutput:
        layer_outputs = []
        layer_predictions = []
        x = self.feature_extractor(x)

        for layer in self.torso:
            if layer[0].downsample:
                layer_predictions = [
                    layer[0].downsample(x)
                    for x in layer_predictions
                ]
            for sublayer in layer:
                x = sublayer(x)
                layer_outputs.append(x)
                layer_predictions.append(x)
        layer_predictions = [
            self.head(x)
            for x in layer_predictions
        ]

        return VariableLengthClassifierOutput(
            layer_activations=layer_outputs,
            layer_predictions=layer_predictions,
        )
    
    @property
    def layers(self) -> List[Tuple[int, int]]:
        return [
            (i, j) for i, layer in enumerate(self.torso) for j in range(len(layer))
        ]
    
    @staticmethod
    def from_pretrained(model_name: str) -> ReducedLengthModelForImageClassification:
        ...