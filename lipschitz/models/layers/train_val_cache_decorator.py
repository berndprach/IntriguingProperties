
from typing import Callable

from torch import nn


def train_val_cached(method: Callable):
    """
    Caches the computation as long as is_training is False.
    As soon as is_training is True, the cache is invalidated.
    """
    return TrainValCacheDescriptor(method)


class CachedMethod:
    def __init__(self, method, instance):
        self.method = method
        self.instance = instance
        self.cache = None

    def __call__(self, *args, **kwargs):
        if self.cache is None:
            self.cache = self.method(self.instance, *args, **kwargs)
        return self.cache

    def invalidate(self):
        self.cache = None


class TrainValCacheDescriptor:
    """
    Caches the computation as long as is_training is False.
    As soon as is_training is True, the cache is invalidated.

    See:
    https://docs.python.org/3/howto/descriptor.html#properties

    """
    def __init__(self, method):
        self.method = method

    def __set_name__(self, owner, name):
        self.private_method_name = "_train_val_cached_" + name

    def __get__(self, instance: nn.Module, owner):
        if instance is None:
            raise AttributeError("This decorator is for methods only!")

        if not hasattr(instance, self.private_method_name):
            cached_method = CachedMethod(self.method, instance)
            setattr(instance, self.private_method_name, cached_method)

        cached_method = getattr(instance, self.private_method_name)
        if instance.training:
            cached_method.invalidate()
            return self.method.__get__(instance, owner)
        else:
            return cached_method
