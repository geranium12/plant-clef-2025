# plant-clef-2025

![Banner Image of PlantCLEF 2025](./docs/banner.png)

This repository contains our code for the [2025 PlantCLEF Kaggle Challenge](https://www.kaggle.com/competitions/plantclef-2025/overview)

A paper acompaniing this repository is available under the title ***"Multi-Label Plant Species Prediction with Metadata-Enhanced Multi-Head Vision Transformers"***.

## Abstract

<!-- TODO -->

## Project Structure

- [`pipeline.py`](./pipeline.py): Main training and inference pipeline.
- [`config/`](./config/): Configuration files for the pipeline.
- [`src/`](./src/): The main source code for the project.
- [`test/`](./test/): The main test code for the project.
- [`predictions/`](./predictions/README.md): Scripts related to prediction.
- [`submissions/`](./submissions/README.md): Scripts related to evaluation and comparison of submissions.
- [`utils/`](./utils/README.md): Utility scripts
- [`notebooks/`](./notebooks/README.md): Jupyter notebooks for exploration and analysis.
- [`docs/`](./docs/README.md): Documentation and information gathered during the project.
- [`misc/`](./misc/README.md): Miscellaneous files and scripts.

## Usage

First, install the dependencies using uv.

```bash
pipx install uv
uv sync
```

Then, set `CUDA_VISIBLE_DEVICES` and `main_process_port` to a random free port (if you run multiple accelerate pipelines) and run the code using uv.

```bash
CUDA_VISIBLE_DEVICES=2,3 uv run accelerate launch --main_process_port=29523 pipeline.py
```

You can specify the run configuration using [`config/config.yaml`](./config/config.yaml).

## Contributing

To contribute you need to install pre-commit hooks in your git repository.

```bash
uv run pre-commit install
```

To run the pre-commit hooks manually, use the following command.

```bash
uv run pre-commit run --all-files
```

## Citation

<!-- TODO -->
