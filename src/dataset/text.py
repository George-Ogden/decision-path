from datasets import load_dataset

from .base import DatasetBuilder


class RawMNLI(DatasetBuilder):
    """Load raw MNLI dataset."""
    @classmethod
    def raw_mnli(cls):
        return load_dataset("glue", "mnli")

@DatasetBuilder.register("mnli")
class MNLIDatasetBuilder(RawMNLI):
    """Load MNLI dataset."""
    @classmethod
    def build(cls):
        return cls.raw_mnli()["validation_matched"]

@DatasetBuilder.register("mnli-mm")
class MNLI_MMDatasetBuilder(RawMNLI):
    """Load MNLI mismatched dataset."""
    @classmethod
    def build(cls):
        return cls.raw_mnli()["validation_mismatched"]

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