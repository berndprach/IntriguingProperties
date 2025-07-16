
UNITS = [(1e9, "GB"), (1e6, "MB"), (1e3, "KB")]


def format_memory(b: float):
    for count, suffix in UNITS:
        if b >= count:
            return f"{b / count:.2f} {suffix}"
    return f"{bytes} B"
