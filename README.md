# CRAR-PyTorch

I replicate the work from Combined Reinforcement Learning via Abstract Representations (CRAR) [\[François-Lavet, Vincent, et al, 2019\]](https://arxiv.org/abs/1809.04506).

![Maze with goal](experiments/random_exploring.gif)

## Installation and Setup

I recommend setting up a [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for running this project. The code should work for all Python 3.6+. Then install CUDA Toolkit (the version depends on you GPU) on your machine, either using `conda`, your package manager, or from [Nvidia's website](https://developer.nvidia.com/cuda-toolkit). Next, install PyTorch (>= 1.4.0), PyTorch Lightning (>=0.7.5), and NumPy (>=1.18) using `pip install -r requirements.txt` or manually using your package manager or `conda`.

Finally, run `python setup.py install` while at the root of this repository.

## Reproducing Labyrinth plot

To reproduce the labyrinth abstract representation plots, change the `plot_dir` under `SimpleMaze-v0` in `config.yaml` to a desired path. Then run `python trainer.py --env SimpleMaze-v0`. Here's my result for the labyrinth plot with the interpretation loss:



![My result](plots/simple_maze_v0/my_plot_wo_interp.pdf)

![Original result](simple_maze_v0/crarrepnointerp.png)
