from pathlib import Path
from typing import Any, Iterable, Union

import yaml

RESULTS_FOLDER = Path("outputs", "results")

Result = dict[str, Union[str, int, float, list]]


def result_from_file(file_path: Union[str, Path]) -> Result:
    file_path = Path(file_path)
    with open(file_path, "r") as f:
        result = yaml.safe_load(f)
    result = flatten(result)
    result["file_path"] = str(file_path)
    return result


def get_results() -> list[Result]:
    results = []
    for fp in sorted(all_filepaths(RESULTS_FOLDER)):
        try:
            result = result_from_file(fp)
        except Exception as e:
            print(f"Error loading {fp}.")
            raise e
        results.append(result)
    print(f"Loaded {len(results)} results.")
    return results


def all_filepaths(ancestor: Path) -> Iterable[Path]:
    for child in ancestor.iterdir():
        if child.is_dir():
            yield from all_filepaths(child)
        else:
            yield child


def flatten(d: dict) -> Result:
    # {"a": {"b": 2, "c": 3}} -> {"a.b": 2, "a.c": 3}
    return {key: value for key, value in _flat_items(d)}


def _flat_items(d: dict) -> Iterable[tuple[str, Any]]:
    for key, value in d.items():
        if isinstance(value, dict):
            for sub_key, sub_value in _flat_items(value):
                yield f"{key}.{sub_key}", sub_value
        else:
            yield key, value


def filter_results(results, constraints, keys: Iterable[str] = tuple()):
    for key, target in constraints.items():
        results = [r for r in results if similar(r.get(key), target)]
        print(f"Filtered to {len(results)} results by {key} = {target}.")

    for key in keys:
        results = [r for r in results if key in r.keys()]
        print(f"Filtered to {len(results)} results with keys {key}.")

    return results


def similar(a: Any, b: Any, threshold: float = 0.01) -> bool:
    """Check if two numbers are similar within a threshold."""
    if a == b:
        return True
    if isinstance(a, float) and isinstance(b, float):
        ratio = a / b
        return 1 - threshold < ratio < 1 + threshold
    return False
