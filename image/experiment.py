import torch.utils.data as data
import torch

from tqdm import tqdm
import argparse

from model import MODELS, VariableLengthResNet
from data import ImageDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", "-b", type=int, default=16)
    parser.add_argument("--num_workers", "-j", type=int, default=4)
    parser.add_argument("--model", "-m", type=int, default=50)
    return parser.parse_args()


def main(args: argparse.Namespace):
    batch_size = args.batch_size
    num_workers = args.num_workers
    model_size = args.model

    dataset = ImageDataset()

    model, weights = MODELS[model_size]
    model = VariableLengthResNet(model(weights="DEFAULT")).cuda()
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
