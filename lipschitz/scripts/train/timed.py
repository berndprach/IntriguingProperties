import argparse
import os
import sys
from pathlib import Path

from .configurations.lipschitz_network import DEFAULT_CONFIGURATION
from lipschitz.scripts.basic.time_batch import time_batch
from lipschitz import data
from lipschitz.io_functions import save_results, state_dict
from lipschitz.io_functions.parser import dictionary_str
from lipschitz.training.evaluate import evaluate
from lipschitz.training.trainer import Trainer

parser = argparse.ArgumentParser()
add_arg = parser.add_argument
add_arg("-u", "--updates", type=dictionary_str, default="{}")
add_arg("-t", "--time-in-hours", type=float, default=1.0)
add_arg("--save-as", default="None")
add_arg("-exp", "--experiment-name", default=None)  # Used for loading results.
add_arg("-s", "--split", default=None)  # Only used for loading results.
add_arg("--time-batch", action="store_true", default=False)

BATCH_TIME_PATH = os.path.join("outputs", "batch_times.txt")


def load_batch_time(c) -> float:
    batch_time = None
    for line in open(BATCH_TIME_PATH, "r"):
        update_str, time_str = line.strip().split(" -> ")
        updates = dictionary_str(update_str)
        line_config = DEFAULT_CONFIGURATION.copy().update(*updates.items())
        if (
                line_config["dataset"] == c["dataset"]
                and line_config["model"] == c["model"]
                # and line_config["augmentation"] == c["augmentation"]
                and line_config["preprocessing"] == c["preprocessing"]
        ):
            batch_time = float(time_str)
    if batch_time is None:
        raise ValueError(f"No batch time found for configuration\n{c}.")
    print(f"Batch time: {batch_time} seconds")
    return batch_time


def timed_train(*args):
    a = parser.parse_args(args)
    print(f"Arguments: {a}")

    if a.time_batch:
        print("Timing batch time.")
        time_batch(a.updates)
        print("Finished timing batch time.")

    c = DEFAULT_CONFIGURATION.copy().update(*a.updates.items())
    print(c)

    bt = load_batch_time(c)
    total_steps = int(a.time_in_hours * 3600 / bt)
    print(f"Training for {a.time_in_hours} hours = {total_steps:,} steps.")

    dl = data.get_loader(c["dataset"], c["preprocessing"])
    dl.train.cycle(total_steps)

    t = Trainer.from_config(c, total_steps)

    t.model.train()
    t.train_on(dl, print_progress=True)

    output = {
        "command": " ".join(sys.argv),
        "arguments": vars(a),
        "configuration": c.as_dict,
        "filename": Path(__file__).name,
        "results": evaluate(t.model, dl, **c["evaluation"]),
    }
    fp = save_results(output)

    if a.save_as != "None":
        state_dict.save(a.save_as, t.model.state_dict(), fp)

