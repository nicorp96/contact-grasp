import os
import numpy as np
import matplotlib.pyplot as plt


def calculate_avg(T, val):
    n = min(
        T.max(),
        len(val),
    )
    cs = np.cumsum(val)
    smoothed_val_part1 = cs[:n] / np.arange(1, n + 1)
    smoothed_val_part2 = ((cs[n:] - cs[:-n]) / n) if len(cs) > n else []
    return T, np.concatenate([smoothed_val_part1, smoothed_val_part2])


def read_numpy_files(path):
    npz_file = np.load(path)
    steps = npz_file["steps"]
    val = npz_file["val"]
    return steps, val


def plot_metrics(steps, val):
    plt.plot(steps, val)
    plt.grid()


def show_plot_metrics(path_1, path_2):
    path_metrics = os.path.join(path_1, path_2, "metrics")
    for name in os.listdir(path_metrics):
        path_m = os.path.join(path_metrics, name)
        steps, val = read_numpy_files(path_m)
        print(name)
        print(f"max = {val.max()}")
        print(f"mean = {val.mean()}")
        plt.gca().set_title(name)
        plot_metrics(
            steps,
            val,
        )
        # steps, val = calculate_avg(steps, val)
        # plot_metrics(steps, val)
        plt.show()


if __name__ == "__main__":
    base = os.path.join(os.getcwd(), "evaluation_data")
    contact_grasp = os.path.join(base, "contact_grasp_point_net")
    contact_grasp_fusion = os.path.join(base, "contact_grasp_point_net_fusion")
    for name in os.listdir(contact_grasp):
        print(name)
        if "bce_2_heads_dl_2_heads" not in name:
            continue
        path_m = os.path.join(contact_grasp, name)
        show_plot_metrics(contact_grasp, path_m)
