from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models import ResNet34_Weights, resnet34
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models import ResNet101_Weights, resnet101
from torchvision.models import ResNet152_Weights, resnet152
from torchvision.models import ResNet

from .base import VariableLengthClassifierOutput, VariableLengthModelForClassification

@VariableLengthModelForClassification.register("resnet")
class ReducedLengthResNetForImageClassification(VariableLengthModelForClassification):
    MODELS = {
        "resnet18": (resnet18, ResNet18_Weights),
        "resnet34": (resnet34, ResNet34_Weights),
        "resnet50": (resnet50, ResNet50_Weights),
        "resnet101": (resnet101, ResNet101_Weights),
        "resnet152": (resnet152, ResNet152_Weights),
    }
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
    
    @classmethod
    def _from_pretrained(cls, model_name: str) -> ReducedLengthResNetForImageClassification:
        model, weights = ReducedLengthResNetForImageClassification.MODELS[model_name]
        return cls(model(weights="DEFALT"))
    