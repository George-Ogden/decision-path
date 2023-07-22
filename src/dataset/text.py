from datasets import load_dataset

from .base import DatasetBuilder


class RawMNLI(DatasetBuilder):
    @classmethod
    def raw_mnli(cls):
        return load_dataset("glue", "mnli")

@DatasetBuilder.register("mnli")
class MNLIDatasetBuilder(RawMNLI):
    @classmethod
    def build(cls):
        return cls.raw_mnli()["validation_matched"]

@DatasetBuilder.register("mnli-mm")
class MNLI_MMDatasetBuilder(RawMNLI):
    @classmethod
    def build(cls):
        return cls.raw_mnli()["validation_mismatched"]