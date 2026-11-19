"""Command-line entry point. Try: python -m voronoi_lab --help"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np

from .geometry import Domain, uniform_sites
from .visuals import VoronoiSimulation, PHASES


def load_csv(path: str) -> np.ndarray:
    """CSV with optional x,y header; ignore additional named columns."""
    with open(path, newline='', encoding='utf-8-sig') as file:
        rows = list(csv.reader(file))
    if not rows:
        raise ValueError('CSV is empty')
    labels = [item.strip().lower() for item in rows[0]]
    if 'x' in labels and 'y' in labels:
        ix, iy = labels.index('x'), labels.index('y')
        data = rows[1:]
    else:
        ix, iy = 0, 1
        data = rows
    points = [[float(row[ix]), float(row[iy])] for row in data if row]
    return np.array(points, dtype=float)


def save_geometry_json(path: str, simulation: VoronoiSimulation, seed: int):
    data = {
        'description': 'Euclidean Voronoi cells clipped to a rectangular viewport',
        'algorithm': 'Convex polygon clipping against all pairwise perpendicular-bisector half-planes',
        'seed': seed,
        'domain': {'width': simulation.domain.width, 'height': simulation.domain.height},
        'focus': simulation.focus,
        'circle_radius': simulation.circle_radius,
        'cells': [
            {'id': i,
             'site': [float(x) for x in simulation.points[i]],
             'vertices_ccw': [[float(x) for x in row] for row in cell],
             'neighbors': simulation.neighbors[i]}
            for i, cell in enumerate(simulation.cells)
        ],
        'checks': simulation.stats,
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2) + '\n')


def parser():
    p = argparse.ArgumentParser(
        prog='python -m voronoi_lab',
        description='Animate a Voronoi diagram built from clipped perpendicular bisectors.')
    p.add_argument('--count', type=int, default=18, help='Number of random uniform sites (default: 18)')
    p.add_argument('--seed', type=int, default=42, help='Reproducible RNG seed (default: 42)')
    p.add_argument('--width', type=float, default=10., help='Domain width (default: 10)')
    p.add_argument('--height', type=float, default=7.5, help='Domain height (default: 7.5)')
    p.add_argument('--margin', type=float, default=.07, help='Uniform point inset fraction (default: .07)')
    p.add_argument('--points-file', type=str, help='Use a CSV with x,y coordinates instead of random sites')
    p.add_argument('--focus', type=int, help='Index of the highlighted site (default: auto)')
    p.add_argument('--circle-radius', type=float, help='Circle radius in domain coordinates (default: auto)')
    p.add_argument('--output', default='voronoi.mp4',
                   help='Output animation (.mp4/.gif) or static figure (.png/.svg/.pdf)')
    p.add_argument('--stage', choices=[phase.key for phase in PHASES], default='done',
                   help='Phase used for a still image (default: done)')
    p.add_argument('--fps', type=int, default=12, help='Frames per second (default: 12)')
    p.add_argument('--dpi', type=int, default=105, help='Output resolution (default: 105)')
    p.add_argument('--speed', type=float, default=1., help='Animation playback speed (default: 1)')
    p.add_argument('--geometry-json', help='Also write sites, polygons and neighbor lists to JSON')
    p.add_argument('--storyboard', help='Also render a nine-panel storyboard PNG')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    domain = Domain(args.width, args.height)
    points = (load_csv(args.points_file) if args.points_file
              else uniform_sites(args.count, domain, seed=args.seed, margin=args.margin))
    sim = VoronoiSimulation(points, domain, focus=args.focus, circle_radius=args.circle_radius)
    suffix = Path(args.output).suffix.lower()
    if suffix in ('.mp4', '.webm', '.gif'):
        sim.save_animation(args.output, fps=args.fps, dpi=args.dpi, speed=args.speed)
    elif suffix in ('.png', '.svg', '.pdf'):
        sim.save_snapshot(args.output, phase=args.stage, dpi=args.dpi)
    else:
        raise SystemExit('Unsupported output extension; choose .mp4, .gif, .png, .svg or .pdf')
    if args.geometry_json:
        save_geometry_json(args.geometry_json, sim, args.seed)
    if args.storyboard:
        sim.save_storyboard(args.storyboard)
    print(f'Saved: {args.output}')
    if args.geometry_json:
        print(f'Geometry: {args.geometry_json}')
    if args.storyboard:
        print(f'Storyboard: {args.storyboard}')
    print(f'Uniform seed: {args.seed} | sites: {len(points)} | highlighted site: {sim.focus}')
    print(f'Neighbors for highlighted site: {sim.local_neighbors}')
    print(f'Domain coverage area error: {sim.stats["area_error"]:.3g}')


if __name__ == '__main__':
    main()
