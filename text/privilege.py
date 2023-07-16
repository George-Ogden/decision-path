from transformers import (
    default_data_collator,
    AutoTokenizer,
    Trainer,
)
from transformers.trainer_pt_utils import find_batch_size
from datasets import load_dataset

from tqdm import tqdm
import functools
import argparse
import torch
import json

from utils import compute_metrics, preprocess_function
from models import ReducedLengthBert


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "-m", type=str, default="bert-base-uncased")
    parser.add_argument("--activation_threshold", "-t", type=float, default=6.)
    return parser.parse_args()


def main(args: argparse.Namespace):
    model_name = args.model_name
    threshold = args.activation_threshold

    model = ReducedLengthBert.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    trainer = Trainer(
        model=model,
        eval_dataset="mnli",
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=default_data_collator,
    )

    raw_datasets = load_dataset(
        "glue",
        "mnli",
    ).map(functools.partial(preprocess_function, tokenizer), batched=True)

    tasks = ["mnli", "mnli-mm"]
    eval_datasets = [
        raw_datasets["validation_matched"],
        raw_datasets["validation_mismatched"],
    ]
    combined = {}

    for eval_dataset, task in zip(eval_datasets, tasks):
        outliers = 0.
        total = 0

        dataloader = trainer.get_eval_dataloader(eval_dataset)
        wrapped_model = trainer._wrap_model(trainer.model, training=False, dataloader=dataloader)
        wrapped_model.eval()

        for batch in tqdm(dataloader):
            total += find_batch_size(batch)
            inputs = trainer._prepare_inputs(batch)
            with torch.no_grad():
                outputs = wrapped_model(**inputs, output_hidden_states=True)
                # L x [B, N, H]
                hidden_states = torch.stack(outputs.hidden_states, axis=1)
                # [B, L, N, H]
                activations = (hidden_states > threshold).float().mean(dim=(2, 3))
                # [B, L]
                outliers += activations.sum(dim=0).cpu().numpy()
        
        # convert to list for json serialization
        combined[task] = (outliers / total).tolist()

    with open(f"results_{model_name}.json", "w") as f:
        json.dump(combined, f)

if __name__ == "__main__":
    args = parse_args()
    main(args)
