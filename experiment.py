from transformers import Trainer, TrainingArguments, default_data_collator
from transformers.trainer_pt_utils import find_batch_size
from tqdm import tqdm
import torch

import argparse
import inspect
import json
import os

from typing import Dict

from src.models import VariableLengthModelForClassification
from src.metrics import METRICS, Metric
from src.dataset import DATASET_BUILDERS

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "-m", type=str, default="bert-base-cased")
    parser.add_argument("--metrics", "-me", nargs="+", type=str, default=["outliers", "kurtosis", "rotated-kurtosis"])
    parser.add_argument("--datasets", "-d", nargs="+", type=str, default=["mnli", "mnli-mm"])
    parser.add_argument("--output-dir", "-o", type=str, default="results", help="Output directory for results")
    parser.add_argument("--threshold", "-t", type=float, default=6.)
    parser.add_argument("--batch_size", "-b", type=int, default=128)
    parser.add_argument("--num_workers", "-j", type=int, default=0)
    return parser.parse_args()

def main(args: argparse.Namespace):
    model_name = args.model_name
    dataset_names = args.datasets
    metric_names = args.metrics
    output_dir = args.output_dir
  
    model = VariableLengthModelForClassification.from_pretrained(model_name).eval().to(device)
    datasets = {
        name: DATASET_BUILDERS[name].build()
        for name in dataset_names
    }

    training_args = TrainingArguments(
        per_device_eval_batch_size=args.batch_size,
        dataloader_num_workers=args.num_workers,
    )

    trainer = Trainer(
        model=model,
        data_collator=default_data_collator,
        args=training_args,
    )

    results = {}
    for dataset_name, dataset in datasets.items():
        dataset = dataset.map(model.preprocess, batched=True)
        dataloader = trainer.get_eval_dataloader(dataset)

        metrics: Dict[str, Metric] = {}
        for metric_name in metric_names:
            metric_type = METRICS[metric_name]
            required_parameters = inspect.signature(metric_type.__init__).parameters
            parameters = {}
            for name in required_parameters:
                if name == "self":
                    continue
                parameters[name] = args.__dict__[name]
            metrics[metric_name] = metric_type(**parameters)
        
        for batch in tqdm(dataloader):
            inputs = trainer._prepare_inputs(batch)
            with torch.no_grad():
                labels = inputs.pop("labels")
                model_output = model(**inputs)
                inputs["labels"] = labels
                inputs["batch_size"] = find_batch_size(batch)
                for metric in metrics.values():
                    metric.update(
                        inputs,
                        model_output
                    )

        results[dataset_name] = {
            name: metric.compute().tolist()
            for name, metric in metrics.items()
        }
    results["layers"] = model.layers

    model_name = model_name.split("/")[-1]
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    with open(f"{output_dir}/{model_name}.json", "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    args = parse_args()
    main(args)
