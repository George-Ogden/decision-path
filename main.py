from torchvision.models import ResNet, ResNet50_Weights, resnet50
import torch.utils.data as data
import torch.nn as nn
import torch

from tqdm import tqdm
from PIL import Image
import argparse

from glob import glob
import os

from classes import IMAGENET2012_CLASSES
from typing import Any, Dict

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", "-b", type=int, default=16)
    parser.add_argument("--num_workers", "-j", type=int, default=4)
    return parser.parse_args()


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, model: ResNet):
        super().__init__()
        for name, module in model.named_children():
            setattr(self, name, module)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # taken from https://github.com/pytorch/vision/blob/71968bc4afb8892284844a7c4cbd772696f42a88/torchvision/models/resnet.py#L266
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


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

    dataset = ImageDataset()

    model = ResNetFeatureExtractor(resnet50(weights="DEFAULT")).cuda()
    transforms = ResNet50_Weights.DEFAULT.transforms()

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
    top_1 = 0.0
    top_5 = 0.0
    for batch in tqdm(dataloader):
        inputs = torch.stack(
            [transforms(image.convert("RGB")) for image in batch["image"]]
        )

        with torch.no_grad():
            predictions = model(inputs.cuda())
            predictions = torch.topk(predictions, k=5, dim=1)[1].cpu()
            top_1 += (predictions[:, 0] == batch["label"]).sum().item()
            top_5 += (predictions == batch["label"].unsqueeze(-1)).sum().item()
    top_1 /= len(dataset)
    top_5 /= len(dataset)
    print(f"Top-1 Accuracy: {top_1}")
    print(f"Top-5 Accuracy: {top_5}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
