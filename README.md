# Voronoi Lab: from ambiguous circles to a non-overlapping partition

A **runnable Python simulation** for constructing 2D Voronoi cells using the actual geometry. It generates a short video or GIF; still figures (PNG/SVG/PDF); a nine-frame storyboard; and machine-readable JSON of every region. All rendering is separate from the geometry, so you can modify the algorithm without changing the animation code.

## Start in one minute

```bash
python -m pip install -r requirements.txt
python -m voronoi_lab --count 18 --seed 42 --output example.mp4
```

For a GIF without FFmpeg:

```bash
python -m voronoi_lab --count 18 --seed 42 --fps 10 --output example.gif
```

For a vector diagram you can use in a paper or slide:

```bash
python -m voronoi_lab --count 35 --seed 9 --output final.svg
```

For **your own points**, provide a CSV with `x,y` columns:

```csv
x,y
1.2,2.0
3.1,5.5
7.7,3.5
```

```bash
python -m voronoi_lab --points-file points.csv --width 10 --height 7.5 \
  --focus 1 --output my-points.mp4 --geometry-json my-points.json
```

The coordinates must fall within `[0,width] x [0,height]`. Random sites use **independent uniform distributions for x and y**, never Gaussian sampling; you can select the reproducible seed and a margin near the boundary.

## The ten animation phases

The animation shows the point cloud, expanding fixed-radius disks, an explicitly highlighted circle-circle overlap, removal of disks, one selected site, connections to true Voronoi-adjacent sites, perpendicular bisectors, half-plane intersections that reduce the selected region, outward revelation of remaining cells, and the completed non-overlapping partition.

The animation is approximately 18 seconds at the default speed. Use `--speed 1.5` for a faster version, `--fps 24` for smoother video, or `--dpi 150` for a larger export. For a preview of individual moments, pass `--storyboard storyboard.png`.

## What the geometry actually does

Given one site `a` and a competitor site `b`, the points that belong on `a`'s side satisfy:

```text
||x - a||^2 <= ||x - b||^2
(b - a) dot x <= (||b||^2 - ||a||^2) / 2
```

The boundary is exactly the **perpendicular bisector** of the segment from `a` to `b`. Begin with the rectangle. Clip it against this half-plane for **every other point**. The intersection is a convex polygon, the Voronoi cell of `a`, cropped to the display rectangle. Repeat for each site.

Crucial distinction: the result is not a polygon of *minimum surface area* obtained by optimization. It is the **intersection of distance-based half-planes**. This produces the unique nearest-site partition (apart from ties on shared boundaries, which have zero area).

The lines connecting sites that share a Voronoi edge are **Delaunay edges**. These are real neighbors: they're not necessarily equal to the `k` closest points, for any fixed `k`. This project identifies neighbors from the completed cell edges using the exact equal-distance property; the focus-point animation uses that true adjacency list.

The rectangle clips infinite Voronoi cells, which would otherwise extend forever at the perimeter. The union of the clipped cells covers the rectangle; their interiors do not overlap.

### Code map

| File / function | Purpose |
|---|---|
| `voronoi_lab/geometry.py :: uniform_sites` | Independent, flat uniform random site generation |
| `geometry.py :: bisector_halfplane` | Derive a midpoint perpendicular bisector |
| `geometry.py :: clip_to_halfplane` | Implement **Sutherland-Hodgman convex polygon clipping** |
| `geometry.py :: compute_cells` | Intersect all pairwise closer-to-site half-planes |
| `geometry.py :: cell_neighbors` | Recover genuine Voronoi/Delaunay neighboring sites |
| `geometry.py :: partition_checks` | Check area coverage and half-plane constraint satisfaction |
| `voronoi_lab/visuals.py :: VoronoiSimulation` | Render stages, generate videos, static SVG and storyboards |
| `voronoi_lab/cli.py` | Command-line arguments and JSON data export |
| `tests/test_geometry.py` | Regression tests for bisectors, clipping, partition area and adjacency |

### Your first algorithm change

Use the geometry API in a notebook or another Python module:

```python
from voronoi_lab.geometry import Domain, uniform_sites, compute_cells, cell_neighbors

bounds = Domain(width=10, height=7.5)
points = uniform_sites(25, bounds, seed=7)
cells = compute_cells(points, bounds)
adjacent = cell_neighbors(points, cells, bounds)

for i, polygon in enumerate(cells):
    print(f"site {i}: {len(polygon)} vertices, neighbors={adjacent[i]}")
```

The central algorithm is intentionally written in pure NumPy and is easy to alter. For production scale (thousands or millions of sites), a Fortune sweep-line or a scipy.spatial.Voronoi/Delaunay implementation is much faster. However, the simple O(N^2) implementation gives you a transparent, testable foundation.

## CLI options

```bash
python -m voronoi_lab --help
```

Notable flags: `--count`, `--seed`, `--width`, `--height`, `--margin`, `--focus`, `--points-file`, `--circle-radius`, `--output`, `--stage`, `--fps`, `--dpi`, `--speed`, `--geometry-json`, `--storyboard`.

## Test

```bash
python -m pip install pytest
python -m pytest -q
```

Includes checks that the sum of cell areas equals the bounding rectangle area, that every cell remains closer to its own site than any competing site, that every vertex lies in the viewport, and that adjacency is symmetric.

## Files and licensing

This example project is provided for your own experimentation and modification. Runtime dependencies are NumPy, Matplotlib and Pillow; MP4 generation needs an `ffmpeg` executable, while GIF generation uses Pillow directly.

## Quick-edit workbench

If you prefer editing Python rather than CLI flags, open `workbench.py` and change the parameter section at the top. Running `python workbench.py` writes an MP4, SVG and storyboard. `sim.cells[i]` exposes the raw NumPy polygon vertices, and `sim.neighbors[i]` exposes the exact adjacent sites.
