from datasets import load_dataset

from .base import DatasetBuilder


class RawMNLI(DatasetBuilder):
    """Load raw MNLI dataset."""
    @classmethod
    def raw_mnli(cls):
        return load_dataset("glue", "mnli")

@DatasetBuilder.register("wikipedia")
class WikipediaDatasetBuilder(DatasetBuilder):
    """Load Wikipedia dataset."""
    @classmethod
    def build(cls):
        return load_dataset("wikipedia", "20220301.simple")["train"]

@DatasetBuilder.register("pile")
class WikipediaDatasetBuilder(DatasetBuilder):
    """Load Pile dataset."""
    @classmethod
    def build(cls):
        return load_dataset("EleutherAI/pile")["validation"]

@DatasetBuilder.register("boolq")
class BoolQDatasetBuilder(DatasetBuilder):
    """Load BoolQ dataset."""
    @classmethod
    def build(cls):
        text = "{passage}\nQ: (yes/no) is the biggest ocean the pacific\nA: yes\nQ: (yes/no) can penguins fly\nA: no\nQ: (yes/no) {question}\nA:"
        return load_dataset("boolq")["validation"].map(
            lambda x: {
                "text": text.format(**x),
                "label": x["answer"]
            },
            batched=False
        )