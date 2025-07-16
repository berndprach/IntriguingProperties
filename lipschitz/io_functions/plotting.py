from matplotlib import pyplot as plt


plt.rcParams.update({'font.size': 14})


def plot_line(x_values, y_values, alpha=0.5, **kwargs):
    sorted_value = zip(*sorted(zip(x_values, y_values)))
    return plt.plot(*sorted_value, alpha=alpha, **kwargs)

