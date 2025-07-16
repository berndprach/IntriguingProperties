TIME_UNITS = [
    (3600, "hours"), (60, "minutes"), (1, "seconds"), (1e-3, "milliseconds")
]


def format_time(seconds: float, f: str = "4.1f") -> str:
    for count, unit in TIME_UNITS:
        if seconds >= count:
            return f"{seconds / count:{f}} {unit}"
    return f"{seconds:.1e} seconds"
