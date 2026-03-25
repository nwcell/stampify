# stampify

Turn black-and-white artwork into stamp-ready STL models.

Repository: https://github.com/nwcell/stampify

## Install

Run the CLI without installing:

```bash
uvx stampify path/to/image.png
```

Install it as a standalone tool:

```bash
uv tool install stampify
stampify path/to/image.png
```

Add it to another Python project:

```bash
uv add stampify
```

## CLI

Generate the bundled sample stamp:

```bash
uv run stampify sample/xmas-cowboy.jpeg
```

Write to a specific STL path:

```bash
uvx stampify path/to/image.png -o stamp.stl
```

Compare the two geometry modes:

```bash
uvx stampify path/to/image.png --mode vector -o stamp-vector.stl
uvx stampify path/to/image.png --mode voxel --resolution 300 -o stamp-voxel.stl
```

Show all options:

```bash
uvx stampify --help
```

## Web app

Launch the browser app from the repo root with one line:

```bash
uv run --extra web python -m ink_print.webapp
```

The dev server reloads automatically when you edit files.

The web app guides you through:

- Upload artwork.
- Review the generated SVG/vector preview.
- Generate the 3D STL.
- Inspect the result in an interactive 3D viewer, then download it.
- SVG artwork is supported alongside raster images.

## Python API

Use it from Python:

```python
from ink_print import StampOptions, write_stamp

options = StampOptions(mode="vector", size=80, border=2, simplify=0.05)
output_path, mesh = write_stamp("path/to/image.png", options=options)
print(output_path, mesh.extents)
```

## Development

Run from a local checkout:

```bash
uv run stampify sample/xmas-cowboy.jpeg
```

Install the local checkout as a tool:

```bash
uv tool install .
```

Add the local checkout to another project:

```bash
uv add git+https://github.com/nwcell/stampify
```

## Notes

- The repo includes `sample/xmas-cowboy.jpeg` as a sample input.
- `vector` mode is the default and produces smoother, smaller meshes.
- `voxel` mode is still available as a fallback.
- `--resolution 0` keeps the source image resolution.
- `--simplify` and `--min-area` are the main cleanup controls for traced artwork.
- The default stamp mirrors the artwork so the printed impression reads correctly.
- The border is raised by default. Disable it with `--no-raised-border`.

## Release automation

This repo includes:

- `.github/workflows/ci.yml` for tests and build validation on pushes and pull requests.
- `.github/workflows/release.yml` for publishing to PyPI from a GitHub Release via Trusted Publishing.

See `RELEASING.md` for the release process.

## License

MIT. See `LICENSE`.
