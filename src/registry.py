from typing import Callable, Dict, Type, TypeVar

T = TypeVar("T")

class Registry:
    """Registry helper for registering classes."""
    registry: Dict[str, T] = {}
    @classmethod
    def register(cls, key: str) -> Callable[[Type[T]], Type[T]]:
        def decorator(class_: Type[T]) -> Type[T]:
            cls.registry[key] = class_
            return class_
        return decorator