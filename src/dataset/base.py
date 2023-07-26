from __future__ import annotations

from typing import Dict
import datasets
import abc

from ..registry import Registry

class DatasetBuilder(abc.ABC, Registry):
    """Abstract class for building datasets."""
    registry: Dict[str, DatasetBuilder] = {}
    
    @abc.abstractclassmethod
    def build(cls) -> datasets.Dataset:
        ...