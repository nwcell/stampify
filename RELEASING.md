# Releasing

## Before the first public release

1. Confirm the GitHub repository is `https://github.com/nwcell/stampify`.
2. Confirm the package metadata in `pyproject.toml` matches the release version and URLs.
3. On PyPI, add a pending Trusted Publisher for:
   - project name: `stampify`
   - owner/repository: `nwcell/stampify`
   - workflow file: `release.yml`
   - environment: `pypi`

PyPI trusted publishing docs:
- https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/
- https://docs.pypi.org/trusted-publishers/using-a-publisher/

## Cutting a release

1. Update the version in `pyproject.toml`.
2. Commit and tag the release.
3. Create a GitHub Release from that tag.
4. The `release.yml` workflow will build, test, and publish to PyPI.

## Local verification

```bash
uv sync --extra test
uv run pytest
uv build
uvx --from . ink-stamp sample/xmas-cowboy.jpeg
```
