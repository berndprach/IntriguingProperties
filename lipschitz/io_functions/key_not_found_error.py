from typing import Iterable


class KeyNotFoundError(KeyError):
    def __init__(self, key, options: Iterable[str] = None, option_type="Key"):
        msg1 = f"{option_type} \"{key}\" not found.\n"
        msg2 = "Available keys: " + ", ".join(sorted(options))
        super().__init__(msg1 + msg2)
