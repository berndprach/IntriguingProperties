import ast
import random
from typing import Iterable

# Set random seed to true random number (not time-based):
random.seed(random.SystemRandom().randint(0, 1_000_000))


def dictionary_str(dict_str: str) -> dict:
    """
    Python code evaluating to dictionary, including random values.
    Examples:
     - "{'name': 'me', 'age': random.choice([20, 30, 40])}"
     - "{'learning_rate': 10**random.uniform(-3, -1)}"
    Do only use this for command line inputs or trusted sources.
    """
    return eval(dict_str)


def iterable_str(s: str) -> Iterable:
    # e.g. "[1, 2, 3]" or "range(1, 10)" "list(range(1, 10)) + [11]"
    return eval(s)


def tuple_str(tup_str: str) -> tuple:
    # E.g. "(1, 2)"
    return ast.literal_eval(tup_str)
