# Contributing

## Development

Install the project and test dependencies:

```bash
uv sync --extra test
```

Run the test suite:

```bash
uv run pytest
```

Run the CLI against the bundled sample:

```bash
uv run ink-stamp sample/xmas-cowboy.jpeg
```

## Pull requests

- Keep changes focused.
- Add or update tests for behavior changes.
- Update the README when the user-facing CLI or API changes.
