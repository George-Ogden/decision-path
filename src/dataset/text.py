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

@DatasetBuilder.register("boolq")
class BoolQDatasetBuilder(DatasetBuilder):
    """Load BoolQ dataset."""
    @classmethod
    def build(cls):
        return load_dataset("boolq")["validation"]