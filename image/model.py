from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models import ResNet34_Weights, resnet34
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models import ResNet101_Weights, resnet101
from torchvision.models import ResNet152_Weights, resnet152
from torchvision.models import ResNet
import torch.nn as nn
import torch

from typing import Tuple

class VariableLengthResNet(nn.Module):
    def __init__(self, model: ResNet):
        super().__init__()
        for name, module in model.named_children():
            setattr(self, name, module)
        self.layers = [getattr(self, name) for name in model._modules.keys() if name.startswith("layer")]

    def forward(self, x: torch.Tensor, layers: Tuple[int, int]) -> torch.Tensor:
        # taken from https://github.com/pytorch/vision/blob/71968bc4afb8892284844a7c4cbd772696f42a88/torchvision/models/resnet.py#L266
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for i, layer in enumerate(self.layers):
            if i < layers[0]:
                x = layer(x)
            elif i == layers[0] and layers[1] > 0:
                x = layer[:layers[1]](x)
            else:
                x = (layer[0].downsample or nn.Identity())(x)
            

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    def update_layers(self, layers: Tuple[int, int]) -> Tuple[int, int]:
        layers = (layers[0], layers[1] + 1)
        if layers[1] >= len(self.layers[layers[0]]):
            layers = (layers[0] + 1, 0)
        return layers

MODELS = {
    18: (resnet18, ResNet18_Weights),
    34: (resnet34, ResNet34_Weights),
    50: (resnet50, ResNet50_Weights),
    101: (resnet101, ResNet101_Weights),
    152: (resnet152, ResNet152_Weights),
}
