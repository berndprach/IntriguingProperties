from datetime import datetime
from time import time
from typing import Iterable

from lipschitz import models
from lipschitz.io_functions.configuration import Configuration
from lipschitz.io_functions.formatting import format_time
from lipschitz.training import metrics, optimizer, scheduler, augmentation


class Trainer:
    def __init__(self, model=None):
        self.model = model
        self.loss_function = None
        self.optimizer = None
        self.scheduler = None
        self.augment = None

    @classmethod
    def from_config(cls, c: Configuration, total_steps: int):
        t = Trainer()
        t.model = models.load_on_device(**c["model"])
        t.loss_function = metrics.get(**c["loss"])
        t.optimizer = optimizer.get(t.model.parameters(), **c["optimizer"])
        t.scheduler = scheduler.get(t.optimizer, total_steps, **c["scheduler"])
        t.augment = augmentation.get(**c["augmentation"])
        return t

    def full_training_run(self, epochs, data_loader):
        print("\nStarting training.")
        self.model.train()
        for _ in PrintProgressOccasionally(range(epochs)):
            self.train_on(data_loader)

    def train_on(self, data_loader: Iterable, print_progress=False):
        if print_progress:
            data_loader = PrintProgressOccasionally(data_loader)
        for x_batch, y_batch in data_loader:
            self.training_step(x_batch, y_batch)

    def training_step(self, x_batch, y_batch) -> None:
        x_batch = self.augment(x_batch)
        predictions = self.model(x_batch)
        losses = self.loss_function(predictions, y_batch)
        loss = losses.mean()
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()


class PrintProgressOccasionally:
    def __init__(self, iterable: Iterable):
        self.iterable = iterable

    def __iter__(self):
        t_print("Started iterating.")
        start_time = time()
        for i, item in enumerate(self.iterable):
            yield item
            if is_power_of_two(i + 1):
                self.print_progress(i + 1, time() - start_time)
        t_print(f"Finished iterating in {format_time(time() - start_time)}.")

    def __len__(self):
        return len(self.iterable)

    def print_progress(self, s: int, elapsed_time: float):
        t_str = format_time(elapsed_time)
        t_print(f"Completed {s:,} / {self.safe_len_str} steps in {t_str}.")

    @property
    def safe_len_str(self):
        try:
            return f"{len(self):,}"
        except TypeError:
            return "?"


def is_power_of_two(n):
    return (n & (n - 1)) == 0


def t_print(s: str):
    prefix = datetime.now().strftime("%H:%M:%S: ")
    print(f"{prefix}{s}")
