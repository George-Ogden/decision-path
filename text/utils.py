from transformers import (
    EvalPrediction,
)
import evaluate
import numpy as np

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


