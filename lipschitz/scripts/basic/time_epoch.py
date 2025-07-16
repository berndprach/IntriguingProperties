import argparse
import os
import sys
import time

from lipschitz import data
from lipschitz.io_functions.formatting import format_time
from lipschitz.io_functions.parser import dictionary_str
from lipschitz.scripts.train.configurations.lipschitz_network import DEFAULT_CONFIGURATION
from lipschitz.training.trainer import Trainer

DEFAULT_MODEL = "{'name': 'SimpleConvNet'}"

parser = argparse.ArgumentParser()
add_arg = parser.add_argument
add_arg("-u", "--updates", type=dictionary_str, default="{}")
add_arg("-n", "--sample_number", type=int, default=100)

# add_arg("-d", "--dataset", default="CIFAR10")
# add_arg("-ts", "--training-size", type=int, default=None)
# add_arg("-b", "--batch-size", type=int, default=256)
# add_arg("-m", "--model-kwargs", type=dictionary_str, default=DEFAULT_MODEL)
# add_arg("--test", action="store_true")

TIME_PATH = os.path.join("outputs", "epoch_times.txt")


def main():
    a = parser.parse_args()
    print(f"Arguments: {a}")
    time_epoch(a.updates, a.sample_number)


def time_epoch(updates: dict, sample_number: int = 3):
    c = DEFAULT_CONFIGURATION.copy().update(*updates.items())
    print(c)

    dl = data.get_loader(c["dataset"], c["preprocessing"])
    dl.train.repeat(sample_number + 1)

    t = Trainer.from_config(c, sample_number + 1)
    t.train_epoch(dl.train)

    epoch_start_time = time.time()
    epoch_seconds = None
    for epoch in range(1, 3 + 1):
        t.train_epoch(dl.train)
        epoch_seconds = time.time() - epoch_start_time
        print(f"Time for epoch {epoch}: {epoch_seconds:.2f} s")
        epoch_start_time = time.time()

    lines = ["", "Results:"]
    for ep in [1, 3, 10, 30, 100, 300, 1_000, 3_000]:
        line = f"Time for {ep:4d} epochs: {format_time(ep * epoch_seconds)}."
        lines.append(line)

    for line in lines:
        print(line)

    with open(TIME_PATH, "a") as f:
        f.write("\n\n" + " ".join(sys.argv))
        f.write("\n".join(lines) + "\n")
