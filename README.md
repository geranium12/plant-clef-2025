# plant-clef-2025

[2025 PlantCLEF Kaggle Challenge](https://www.kaggle.com/competitions/plantclef-2025/overview)

![Banner Image of PlantCLEF 2025](./docs/banner.png)

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

You can specify the run configuration using `config/config.yaml`.

The notebooks in `predictions/` can be used to merge logits from multiple models.

The notebooks in `submissions/` can be used to analyze similarity between different predictions.

## Contributing

To contribute you need to install pre-commit hooks in your git repository.

```bash
uv run pre-commit install
```

To run the pre-commit hooks manually, use the following command.

```bash
uv run pre-commit run --all-files
```
