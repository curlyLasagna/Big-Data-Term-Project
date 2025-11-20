## Pre-publish Upvote Predictor

A model that predicts how a post's upvotes depending on the subreddit

## Development

### Contributing to the paper

#### Install typst

Install typst via [scoop](https://scoop.sh/#/) through `scoop install main/typst`

#### Compiling document

`typst compile main.typ`

### [devenv](https://devenv.sh/) for system binaries (Optional)

> To install devenv, you will need to have nix installed in your system. See https://determinate.systems/blog/determinate-nix-installer/

### [uv](https://docs.astral.sh/uv/) for Python dependencies

- `uv venv`: Create a virtual environment (Not required if using devenv)
- `uv sync`: Install dependencies listed in `pyproject.toml` 

### Marimo notebook

[Marimo](https://marimo.io/) is the Jupyter alternative that we will be using for our experiments.

After syncing dependencies via `uv sync` and sourcing to the proper venv, you should be able to run `marimo edit main.py` within the `notebook/` directory.

This should open up your default web browser to the notebook
