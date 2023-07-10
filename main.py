from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models import ResNet34_Weights, resnet34
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models import ResNet101_Weights, resnet101
from torchvision.models import ResNet152_Weights, resnet152
from torchvision.models import ResNet
import torch.utils.data as data
import torch.nn as nn
import torch

from tqdm import tqdm
from PIL import Image
import argparse

from glob import glob
import os

from classes import IMAGENET2012_CLASSES
from typing import Any, Dict, Tuple

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", "-b", type=int, default=16)
    parser.add_argument("--num_workers", "-j", type=int, default=4)
    parser.add_argument("--model", "-m", type=int, default=50)
    return parser.parse_args()


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, model: ResNet):
        super().__init__()
        for name, module in model.named_children():
            setattr(self, name, module)
        self.layers = [self.layer1, self.layer2, self.layer3, self.layer4]

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

class ImageDataset(data.Dataset):
    label2id = {label: i for i, label in enumerate(IMAGENET2012_CLASSES)}
    id2label = list(IMAGENET2012_CLASSES)

    def __init__(self):
        self.images = glob("data/*.JPEG")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        filename = self.images[idx]
        image = Image.open(filename)
        cls = os.path.splitext(filename)[0].split("_")[-1]
        return {"image": image, "label": self.label2id[cls]}

    def __len__(self) -> int:
        return len(self.images)


def main(args: argparse.Namespace):
    batch_size = args.batch_size
    num_workers = args.num_workers
    model_size = args.model

    dataset = ImageDataset()

    model, weights = MODELS[model_size]
    model = ResNetFeatureExtractor(
        model(weights="DEFAULT")
    ).cuda()
    transforms = weights.DEFAULT.transforms()

    dataloader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        collate_fn=lambda batch: {
            "image": [image["image"] for image in batch],
            "label": torch.tensor([image["label"] for image in batch]),
        },
    )
    layers = (0, 0)
    while True:
        top_1 = 0.0
        top_5 = 0.0
        for batch in tqdm(dataloader):
            inputs = torch.stack(
                [transforms(image.convert("RGB")) for image in batch["image"]]
            )

            with torch.no_grad():
                predictions = model(inputs.cuda(), layers)
                predictions = torch.topk(predictions, k=5, dim=1)[1].cpu()
                top_1 += (predictions[:, 0] == batch["label"]).sum().item()
                top_5 += (predictions == batch["label"].unsqueeze(-1)).sum().item()
        top_1 /= len(dataset)
        top_5 /= len(dataset)
        print(f"{layers[0]+1}-{layers[1]+1} | {top_1*100:.4f}% | {top_5*100:.4f}%")
        try:
            layers = model.update_layers(layers)
        except IndexError:
            break


if __name__ == "__main__":
    args = parse_args()
    main(args)
