# stampify

Turn black-and-white artwork into stamp-ready STL models.

Repository: https://github.com/nwcell/stampify

## Install

Install from a checkout:

```bash
uv tool install .
```

Run without installing:

```bash
uvx --from . ink-stamp sample/xmas-cowboy.jpeg
```

## Standalone CLI

Run it directly from the repo:

```bash
uv run ink-stamp sample/xmas-cowboy.jpeg
```

Install it as a standalone tool:

```bash
uv tool install .
ink-stamp sample/xmas-cowboy.jpeg
```

## Sample

The repo includes `sample/xmas-cowboy.jpeg` as a sample input. Generate the sample stamp with:

```bash
uv run ink-stamp sample/xmas-cowboy.jpeg -o xmas-cowboy-stamp.stl
```

Or compare the two geometry modes against the same sample:

```bash
uv run ink-stamp sample/xmas-cowboy.jpeg --mode vector -o xmas-cowboy-vector-stamp.stl
uv run ink-stamp sample/xmas-cowboy.jpeg --mode voxel --resolution 300 -o xmas-cowboy-voxel-stamp.stl
```

## Add To Another Project

Add the package from GitHub:

```bash
uv add git+https://github.com/nwcell/stampify
```

Once you publish to PyPI, the same package can be added with:

```bash
uv add stampify
```

Use it from Python:

```python
from ink_print import StampOptions, write_stamp

options = StampOptions(mode="vector", size=80, border=2, simplify=0.05)
output_path, mesh = write_stamp("sample/xmas-cowboy.jpeg", options=options)
print(output_path, mesh.extents)
```

## Notes

- `vector` mode is the default and produces smoother, smaller meshes.
- `voxel` mode is still available as a fallback.
- `--resolution 0` keeps the source image resolution.
- `--simplify` and `--min-area` are the main cleanup controls for traced artwork.

## Release automation

This repo includes:

- `.github/workflows/ci.yml` for tests and build validation on pushes and pull requests.
- `.github/workflows/release.yml` for publishing to PyPI from a GitHub Release via Trusted Publishing.

See `RELEASING.md` for the setup steps you still need to complete in GitHub and PyPI.
