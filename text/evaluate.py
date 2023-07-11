from transformers import (
    default_data_collator,
    AutoTokenizer,
    BertForSequenceClassification,
    EvalPrediction,
    Trainer,
)
from datasets import load_dataset
import evaluate

import numpy as np
import functools
import argparse


class ReducedLengthBert(BertForSequenceClassification):
    def reduce_layers(self, layers: int):
        self.bert.encoder.layer = self.bert.encoder.layer[:layers]
    
    @property
    def num_layers(self):
        return len(self.bert.encoder.layer)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "-m", type=str, default="bert-base-uncased")
    return parser.parse_args()


metric = evaluate.load("glue", "mnli")


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    result = metric.compute(predictions=preds, references=p.label_ids)
    if len(result) > 1:
        result["combined_score"] = np.mean(list(result.values())).item()
    return result


def preprocess_function(tokenizer, examples):
    # Tokenize the texts
    sentence1_key, sentence2_key = "premise", "hypothesis"
    max_seq_length = 128
    return tokenizer(
        examples[sentence1_key],
        examples[sentence2_key],
        padding="max_length",
        max_length=max_seq_length,
        truncation=True,
    )


def main(args: argparse.Namespace):
    model_name = args.model_name

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

    for layers in reversed(range(model.num_layers + 1)):
        model.reduce_layers(layers)
        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)
            metrics["eval_samples"] = len(eval_dataset)

            if task == "mnli-mm":
                metrics = {k + "_mm": v for k, v in metrics.items()}
            combined.update(metrics)

            trainer.log_metrics(f"eval-{layers}", metrics)


if __name__ == "__main__":
    args = parse_args()
    main(args)
