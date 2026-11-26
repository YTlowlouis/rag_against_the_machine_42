ARGS ?=

install:
	uv sync


run:
	uv run python -m src $(ARGS)


debug:
	uv run python -m pdb -m src $(ARGS)


test:
	uv run pytest


clean:
	rm -rf __pycache__ .mypy_cache .pytest_cache
	find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} +


lint:
	uv run flake8 . --extend-exclude .venv,.env
	uv run mypy . --exclude '(^|/)\.(venv|env)/' --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs


lint-strict:
	uv run flake8 . --extend-exclude .venv,.env
	uv run mypy . --exclude '(^|/)\.(venv|env)/' --strict
