import argparse
import os

import torch
from lipschitz.io_functions.formatting import format_time

from lipschitz import data
from lipschitz.data import preprocessors
from lipschitz.io_functions.parser import dictionary_str
from lipschitz.scripts.train.configurations.lipschitz_network import DEFAULT_CONFIGURATION
from lipschitz.training.trainer import Trainer

parser = argparse.ArgumentParser()
add_arg = parser.add_argument
add_arg("-u", "--updates", type=dictionary_str, default="{}")
add_arg("-n", "--sample_number", type=int, default=100)

BATCH_TIME_PATH = os.path.join("outputs", "batch_times.txt")


def main():
    a = parser.parse_args()
    print(f"Arguments: {a}")

    time_batch(a.updates, a.sample_number)


def time_batch(updates: dict, sample_number: int = 100):
    c = DEFAULT_CONFIGURATION.copy().update(*updates.items())
    print(c)

    dl = data.get_loader(c["dataset"], c["preprocessing"])
    dl.train.cycle(sample_number + 1)

    t = Trainer.from_config(c, sample_number+1)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    di = iter(dl.train)

    # Warm the model up for the first batch:
    t.training_step(*next(di))
    torch.cuda.synchronize()

    start_event.record()
    for batch in di:
        t.training_step(*batch)

    end_event.record()
    torch.cuda.synchronize()
    total_seconds = start_event.elapsed_time(end_event) / 1000

    average_seconds = total_seconds / sample_number

    print(f"Average time per batch: {format_time(average_seconds)}")
    with open(BATCH_TIME_PATH, "a") as f:
        f.write(f"{updates} -> {average_seconds}\n")
