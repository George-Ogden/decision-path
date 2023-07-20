from __future__ import annotations

import abc
from typing import Callable, Type


class Metric(abc.ABC):
    registry = {}
    @abc.abstractmethod
    def update(self, *args, **kwargs):
        ...
    
    @abc.abstractmethod
    def compute(self, *args, **kwargs):
        ...
    
    @classmethod
    def register(cls, key: str) -> Callable[[Type[Metric]], Type[Metric]]:
        def decorator(model_class: Type[Metric]) -> Type[Metric]:
            cls.registry.append((key, model_class))
            return model_class
        return decorator
