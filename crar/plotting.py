import pytorch_lightning as pl
import torch
import numpy as np
from crar.environments import SimpleMaze
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib.patches import Circle, Rectangle
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, DrawingArea, HPacker
from typing import Union
from pathlib import Path

matplotlib.use("qt5agg")


def most_recent(buffer, n: Union[None, int] = None):
    if n is None:
        return buffer[0]
    buffer_copy, vals = buffer.copy(), []
    for _ in range(n):
        vals.append(buffer_copy.popleft())

    return vals


def plot_maze_abstract_transitions(
    all_inputs, all_abs_inputs, model, global_step, plot_dir
):
    if not isinstance(plot_dir, Path):
        plot_dir = Path(plot_dir)

    exp_seq = list(reversed(most_recent(model.replay_buffer.buffer, 1000)))

    # matplotlib.rcParams["figure.figsize"] = (15, 15)
    n = 1000
    history = []
    for i, (obs, *_) in enumerate(exp_seq):
        history.append(obs)
    history = np.array(history)
    print(history.shape)

    abstract_states = model.agent.encode(history)
    m = cm.ScalarMappable(cmap=cm.jet)
    x, y = abstract_states.detach().cpu().numpy().T
    # print(x)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(r"$X_1$")
    ax.set_ylabel(r"$X_2$")

    for i in range(n - 1):
        predicted1 = (
            model.agent.compute_transition(
                abstract_states[i : i + 1], torch.as_tensor([0], device="cuda")
            )
            .detach()
            .cpu()
            .numpy()
        )

        # print(f"predicted1 = {predicted1}")

        predicted2 = (
            (
                model.agent.compute_transition(
                    abstract_states[i : i + 1], torch.as_tensor([1], device="cuda")
                )
            )
            .detach()
            .cpu()
            .numpy()
        )

        predicted3 = (
            (
                model.agent.compute_transition(
                    abstract_states[i : i + 1], torch.as_tensor([2], device="cuda")
                )
            )
            .detach()
            .cpu()
            .numpy()
        )

        predicted4 = (
            (
                model.agent.compute_transition(
                    abstract_states[i : i + 1], torch.as_tensor([3], device="cuda")
                )
            )
            .detach()
            .cpu()
            .numpy()
        )

        ax.plot(
            np.concatenate([x[i : i + 1], predicted1[0, :1]]),
            np.concatenate([y[i : i + 1], predicted1[0, 1:2]]),
            color="royalblue",
            alpha=0.75,
        )

        ax.plot(
            np.concatenate([x[i : i + 1], predicted2[0, :1]]),
            np.concatenate([y[i : i + 1], predicted2[0, 1:2]]),
            color="crimson",
            alpha=0.75,
        )

        ax.plot(
            np.concatenate([x[i : i + 1], predicted3[0, :1]]),
            np.concatenate([y[i : i + 1], predicted3[0, 1:2]]),
            color="mediumspringgreen",
            alpha=0.75,
        )

        ax.plot(
            np.concatenate([x[i : i + 1], predicted4[0, :1]]),
            np.concatenate([y[i : i + 1], predicted4[0, 1:2]]),
            color="black",
            alpha=0.75,
        )
    # Plot the dots at each time step depending on the action taken
    length_block = [[0, 18], [18, 19], [19, 31]]
    for i in range(3):
        colors = ["blue", "orange", "green"]
        line3 = ax.scatter(
            all_abs_inputs[length_block[i][0] : length_block[i][1], 0],
            all_abs_inputs[length_block[i][0] : length_block[i][1], 1],
            c=colors[i],
            marker="x",
            edgecolors="k",
            alpha=0.5,
            s=100,
        )
    axes_lims = [ax.get_xlim(), ax.get_ylim()]

    box1b = TextArea(
        " Estimated transitions (action 0, 1, 2 and 3): ", textprops=dict(color="k")
    )
    box2b = DrawingArea(90, 20, 0, 0)
    el1b = Rectangle((5, 10), 15, 2, fc="royalblue", alpha=0.75)
    el2b = Rectangle((25, 10), 15, 2, fc="crimson", alpha=0.75)
    el3b = Rectangle((45, 10), 15, 2, fc="mediumspringgreen", alpha=0.75)
    el4b = Rectangle((65, 10), 15, 2, fc="black", alpha=0.75)
    box2b.add_artist(el1b)
    box2b.add_artist(el2b)
    box2b.add_artist(el3b)
    box2b.add_artist(el4b)

    boxb = HPacker(children=[box1b, box2b], align="center", pad=0, sep=5)

    anchored_box = AnchoredOffsetbox(
        loc=3,
        child=boxb,
        pad=0.0,
        frameon=True,
        bbox_to_anchor=(0.0, 0.98),
        bbox_transform=ax.transAxes,
        borderpad=0.0,
    )
    ax.add_artist(anchored_box)
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_dir / f"plot_{global_step}.pdf")


def plot_simple_abstract_space(global_step, model, plot_dir):
    if not isinstance(plot_dir, Path):
        plot_dir = Path(plot_dir)

    exp_seq = list(reversed(most_recent(model.replay_buffer.buffer, 1000)))
    history = []
    for i, (obs, *_) in enumerate(exp_seq):
        history.append(obs)
    history = np.array(history)
    print(history.shape)

    abstract_states = model.agent.encode(history)
    m = cm.ScalarMappable(cmap=cm.jet)
    x, y = abstract_states.detach().cpu().numpy().T
    # print(x)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(r"$X_1$")
    ax.set_ylabel(r"$X_2$")

    n = 1000

    for i in range(n - 1):
        predicted1 = (
            model.agent.compute_transition(
                abstract_states[i : i + 1], torch.as_tensor([0], device="cuda")
            )
            .detach()
            .cpu()
            .numpy()
        )

        # print(f"predicted1 = {predicted1}")

        predicted2 = (
            (
                model.agent.compute_transition(
                    abstract_states[i : i + 1], torch.as_tensor([1], device="cuda")
                )
            )
            .detach()
            .cpu()
            .numpy()
        )

        # predicted3 = (
        #     (
        #         model.agent.compute_transition(
        #             abstract_states[i : i + 1], torch.as_tensor([2], device="cuda")
        #         )
        #     )
        #     .detach()
        #     .cpu()
        #     .numpy()
        # )

        # predicted4 = (
        #     (
        #         model.agent.compute_transition(
        #             abstract_states[i : i + 1], torch.as_tensor([3], device="cuda")
        #         )
        #     )
        #     .detach()
        #     .cpu()
        #     .numpy()
        # )

        ax.plot(
            np.concatenate([x[i : i + 1], predicted1[0, :1]]),
            np.concatenate([y[i : i + 1], predicted1[0, 1:2]]),
            color="royalblue",
            alpha=0.75,
        )

        ax.plot(
            np.concatenate([x[i : i + 1], predicted2[0, :1]]),
            np.concatenate([y[i : i + 1], predicted2[0, 1:2]]),
            color="crimson",
            alpha=0.75,
        )

        # ax.plot(
        #     np.concatenate([x[i : i + 1], predicted3[0, :1]]),
        #     np.concatenate([y[i : i + 1], predicted3[0, 1:2]]),
        #     color="mediumspringgreen",
        #     alpha=0.75,
        # )

        # ax.plot(
        #     np.concatenate([x[i : i + 1], predicted4[0, :1]]),
        #     np.concatenate([y[i : i + 1], predicted4[0, 1:2]]),
        #     color="black",
        #     alpha=0.75,
        # )

    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_dir / f"plot_{global_step}.pdf")
