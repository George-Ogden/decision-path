from datasets import load_dataset

from .register import register

raw_datasets = load_dataset(
    "glue",
    "mnli",
)

mnli = raw_datasets["validation_matched"]
mnli_mm = raw_datasets["validation_mismatched"]
register("mnli", mnli)
register("mnli-mm", mnli_mm)