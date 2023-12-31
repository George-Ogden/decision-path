from transformers import Trainer, TrainingArguments, default_data_collator
from transformers.trainer_pt_utils import find_batch_size
from tqdm import tqdm
import numpy as np
import torch

import argparse
import inspect
import json
import os

from typing import Dict, List, Tuple, Union

from src.models import VariableLengthModelForPrediction
from src.metrics import METRICS, Metric
from src.dataset import DATASET_BUILDERS

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "-m", type=str, default="EleutherAI/pythia-70m")
    parser.add_argument("--revision", "-r", type=str, default="main")
    parser.add_argument("--metrics", "-me", nargs="+", type=str, default=["rms", "outliers"])
    parser.add_argument("--datasets", "-d", nargs="+", type=str, default=["wikipedia"])
    parser.add_argument("--output-dir", "-o", type=str, default="results", help="Output directory for results")
    parser.add_argument("--threshold", "-t", type=float, default=.5)
    parser.add_argument("--batch_size", "-b", type=int, default=32)
    parser.add_argument("--num_workers", "-j", type=int, default=0)
    return parser.parse_args()

def main(args: argparse.Namespace):
    model_name = args.model_name
    dataset_names = args.datasets
    metric_names = args.metrics
    output_dir = args.output_dir
    revision = args.revision
    
    model = VariableLengthModelForPrediction.from_pretrained(model_name, revision=revision)
    datasets = {
        name: DATASET_BUILDERS[name].build()
        for name in dataset_names
    }

    training_args = TrainingArguments(
        per_device_eval_batch_size=args.batch_size,
        dataloader_num_workers=args.num_workers,
        output_dir=output_dir,
        do_eval=True,
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
                parameters[name] = vars(args)[name]
            metrics[metric_name] = metric_type(**parameters)
        
        for batch in tqdm(dataloader):
            inputs = trainer._prepare_inputs(batch)
            with torch.no_grad():
                labels = None
                if "labels" in inputs:
                    labels = inputs.pop("labels")
                model_output = model(**inputs)
                if labels is not None:
                    inputs["labels"] = labels
                inputs["batch_size"] = find_batch_size(batch)
                inputs["pad_token_id"] = model.tokenizer.pad_token_id
                for metric in metrics.values():
                    metric.update(
                        inputs,
                        model_output
                    )

        results[dataset_name] = {
            name: metric.compute()
            for name, metric in metrics.items()
        }
    
    for token in ["yes", "no"]:
        # Space is important for correct tokenization.
        index = model.tokenizer.encode(" " + token)[-1]
        vector = model.head.weight[index]
        results[token.strip()] = vector.detach().cpu().numpy()
    
    def convert_to_pickle(json: Dict[str, Union[Dict, np.ndarray]], key: Tuple[str, ...] = ()):
        to_store = {}
        filename = os.path.join(output_dir, *key, "data.npz")
        for k, v in json.items():
            if isinstance(v, dict):
                convert_to_pickle(v, key + (k,))
            elif isinstance(v, np.ndarray):
                to_store[k] = v
                json[k] = filename
        if to_store:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            np.savez_compressed(filename, **to_store)
            json[k] = filename

    model_name = model_name.split("/")[-1]
    output_dir = os.path.join(output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)
    convert_to_pickle(results)

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    args = parse_args()
    main(args)
