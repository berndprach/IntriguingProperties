import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

from lipschitz.io_functions.parser import dictionary_str
from lipschitz.scripts.plot.results import get_data

plt.rcParams.update({'font.size': 14})

# RESULTS_FOLDER = Path("outputs", "results")
PLOT_FOLDER = Path("outputs", "result_plots")
PLOT_FOLDER.mkdir(exist_ok=True)

parser = argparse.ArgumentParser()
add_arg = parser.add_argument
add_arg("-c", "--constraints", type=dictionary_str, default="{'epochs': 100}")
add_arg("-s", "--split-by", type=str, default="arguments.model_name")
add_arg("-x", "--x-key", type=str, default="arguments.lr")
add_arg("-xs", "--x-scale", choices=["linear", "log"], default="linear")
add_arg("-y", "--y-key", type=str, default="results.eval.CRA(0.14)")
# add_arg("-ys", "--y-scale", choices=["linear", "log"], default="linear")
add_arg("-u", "--unique-values", action="store_true", default=False)


def main():
    plot_results()


def plot_results():
    args = parser.parse_args()
    print(f"Arguments: {args}")

    data = get_data(
        args.constraints, args.x_key, args.y_key, args.split_by
    )

    yd = {s: {args.x_key: xs, args.y_key: ys} for s, (xs, ys) in data.items()}
    print("\nData for plotting:")
    print(yaml.safe_dump(yd, default_flow_style=None))

    value_to_loc = {}
    w = 0.8 / len(data)
    for i, (split_name, (xs, ys)) in enumerate(data.items()):
        for x in xs:
            if x not in value_to_loc:
                value_to_loc[x] = len(value_to_loc)

        # locations = [value_to_loc[x]+i/len(data)-1/4 for x in xs]
        locations = [value_to_loc[x] - 0.5 + (2*i+1)/(2*len(data)) for x in xs]
        plt.bar(locations, ys, label=split_name, alpha=0.5, width=w)

    plt.title(f"{args.y_key}")
    plt.xlabel(f"{args.x_key} ({args.x_scale})")
    plt.ylabel(args.y_key)

    plt.xscale(args.x_scale)
    plt.xticks(list(value_to_loc.values()), list(value_to_loc.keys()))

    legend_title = " ".join(args.split_by.split(".")[-2:])
    legend_title = legend_title.replace("_", " ").capitalize() + ":"
    plt.legend(title=legend_title)
    plt.grid()

    plt.show()


if __name__ == "__main__":
    main()
