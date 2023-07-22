from .text import MNLIDatasetBuilder, MNLI_MMDatasetBuilder
from .image import ImageNetDatasetBuilder
from .base import DatasetBuilder

DATASET_BUILDERS = DatasetBuilder.registry