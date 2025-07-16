import argparse
import math
import sys
from datetime import datetime
from pathlib import Path
import hashlib
from typing import Any, Iterable

import yaml
import matplotlib.pyplot as plt

from io_functions.result_parsing import all_files, flatten

plt.rcParams.update({'font.size': 14})

RESULTS_FOLDER = Path("outputs", "results")
PLOT_FOLDER = Path("outputs", "result_plots")
PLOT_FOLDER.mkdir(exist_ok=True)

parser = argparse.ArgumentParser()
add_arg = parser.add_argument
add_arg("-c", "--constraints", type=eval, default="{'epochs': 100}")


def get_item(key, result):
    for prefix in ["", "arguments.", "results."]:
        try:
            return result[prefix + key]
        except KeyError:
            continue
    raise KeyError(f"Key {key} not found in result.")


def get(key, result, default=None):
    try:
        return get_item(key, result)
    except KeyError:
        return default


def delete_results(*args):
    a = parser.parse_args(args)
    print(f"Arguments: {args}")

    for fp in sorted(all_files(RESULTS_FOLDER)):
        with open(fp, "r") as f:
            result = yaml.safe_load(f)
        result = flatten(result)

        if all(get(k, result) == v for k, v in a.constraints.items()):
            # print(f"Matched result: {fp}")
            query = f"\nMatched result: {fp}. Delete? (y/n): "
            decision = input(query).strip().lower()
            if decision == "y":
                fp.unlink()
                print(f"Deleted.")
