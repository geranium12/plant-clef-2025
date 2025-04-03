# plant-clef-2025
https://www.kaggle.com/competitions/plantclef-2025/overview

## Contributing

To contribute you need to install pre-commit hooks in your git repository.

```bash
uv run pre-commit install
```

To run the pre-commit hooks manually, use the following command.

```bash
uv run pre-commit run --all-files
```

## Usage

First, install the dependencies using uv.

```bash
pipx install uv
uv sync
```

Then, you can run the code using uv.

```bash
uv run pipeline.py
```
