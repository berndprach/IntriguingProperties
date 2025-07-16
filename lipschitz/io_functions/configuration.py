import copy
from typing import Any

import yaml


NestedDict = dict[str, Any]  # e.g. {"model": {"name": "SimpleConvNet"}}


class Configuration:
    def __init__(self, d: NestedDict):
        self.configuration = d

    def __getitem__(self, key: str) -> Any:
        return self.configuration[key]

    def __setitem__(self, key: str, value: Any):
        _nested_update_single(self.configuration, key, value)

    def __str__(self):
        config_str = yaml.safe_dump(self.configuration).replace("  ", "    ")
        return f"\nConfiguration:\n{config_str}\n"

    @classmethod
    def from_yaml(cls, yaml_str: str):
        return Configuration(yaml.safe_load(yaml_str))

    @property
    def as_dict(self) -> NestedDict:
        return self.configuration

    def copy(self):
        d = copy.deepcopy(self.configuration)
        return Configuration(d)

    def update(self, *updates: tuple[str, Any]):
        """ Make sure to first .copy() default configurations. """
        nested_update(self.configuration, *updates)
        return self  # Update in place and return self for chaining.


def _nested_update_single(d: NestedDict, key: str, v: Any):
    keys = key.split(".")
    for k in keys[:-1]:
        try:
            d = d[k]
        except KeyError:
            raise KeyError(f"Key \"{key}\" not found. (Problem with {k}.)")
    d[keys[-1]] = v


def nested_update(d: NestedDict, *updates: tuple[str, Any]) -> None:
    for k, v in updates:
        _nested_update_single(d, k, v)
