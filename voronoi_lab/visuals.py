"""Render a faithful, explainable Voronoi construction with Matplotlib.

The rendering has no effect on the geometry, making it easy to replace the
cell implementation independently of the teaching animation.
"""
from __future__ import annotations

from dataclasses import dataclass
import colorsys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')   # works on headless systems and over SSH
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle, Polygon
from PIL import Image, ImageOps, ImageDraw

from .geometry import (Domain, compute_cells, cell_neighbors, find_focus,
                       ordered_neighbors, breadth_first_order,
                       clip_closer_to_a, bisector_segment, partition_checks)

# Visual constants can be restyled without modifying any geometric operation.
BACKGROUND = '#f7f9fc'
PLOT_BG = '#ffffff'
INK = '#14263b'
MUTED = '#607386'
GRID = '#e5ebf2'
BLUE = '#1765c1'
DEEP_BLUE = '#12559b'
VIOLET = '#8a4ad5'
RED = '#d64c59'
AMBER = '#ef9d31'
TEAL = '#37a4a0'


@dataclass(frozen=True)
class Phase:
    key: str
    seconds: float
    heading: str
    detail: str
    note: str


PHASES = [
    Phase('points', 1.0, 'Begin with uniformly scattered points',
          'Independent, uniform random samples in a 2D rectangle.',
          'Nothing has been assigned a region yet.'),
    Phase('circles', 1.5, 'A naive approach: draw a circle around each point',
          'Fixed-radius neighborhoods are simple, but ambiguous.',
          'Areas can be inside several circles or none.'),
    Phase('overlap', 1.5, 'Overlapping disks claim the same space',
          'The red lens belongs to both circles.',
          'Which point should own the highlighted overlap?'),
    Phase('clear', 0.7, 'Replace arbitrary circles with a distance rule',
          'We want every location assigned to its nearest point.',
          'One consistent rule resolves the overlap.'),
    Phase('focus', 0.7, 'Consider one site',
          'Construct the region closer to this site than to any other.',
          'The selected site is highlighted in blue.'),
    Phase('links', 2.1, 'Connect the site to its Voronoi neighbors',
          'These links form the local Delaunay adjacency graph.',
          'Not every k-nearest site is necessarily a cell neighbor.'),
    Phase('cuts', 4.4, 'Bisect each connection and keep the closer side',
          'Each perpendicular bisector contributes a half-plane.',
          'Intersect those half-planes to form one convex cell.'),
    Phase('cell', 1.2, 'The site now owns a unique polygon',
          'The blue polygon is the intersection of all constraints.',
          'Each edge is equally distant from two neighboring sites.'),
    Phase('fill', 3.0, 'Apply the same construction to the other sites',
          'Cells expand outward in adjacency order for this animation.',
          'The construction itself is independent per site.'),
    Phase('done', 2.2, 'One region per point. No overlapping interiors.',
          'Together, the cells tile the entire displayed rectangle.',
          'This is a Voronoi diagram (clipped to the viewport).'),
]


def _ease(t: float) -> float:
    t = float(np.clip(t, 0, 1))
    return t * t * (3 - 2 * t)


def _mix(a, b, u):
    return (1 - u) * np.asarray(a) + u * np.asarray(b)


def circle_overlap_polygon(a: np.ndarray, b: np.ndarray,
                           radius: float, steps: int = 65) -> np.ndarray:
    """Exact circle-circle lens boundary sampled along its two circular arcs."""
    vector = b - a
    distance = float(np.linalg.norm(vector))
    if not (1e-12 < distance < 2 * radius):
        return np.empty((0, 2))
    direction = float(np.arctan2(vector[1], vector[0]))
    angle = float(np.arccos(np.clip(distance / (2 * radius), -1, 1)))
    first_angles = np.linspace(direction - angle, direction + angle, steps)
    second_angles = np.linspace(direction + np.pi - angle,
                                direction + np.pi + angle, steps)
    arc1 = a + radius * np.c_[np.cos(first_angles), np.sin(first_angles)]
    arc2 = b + radius * np.c_[np.cos(second_angles), np.sin(second_angles)]
    return np.vstack([arc1, arc2])


def _select_colors(neighbors: list[list[int]], focus: int) -> list[str]:
    """Greedy adjacency color separation, then pastelize for filled cells."""
    palette = ['#b9e3fc', '#f7d6d4', '#dbefcc', '#fbe9bb', '#e5dafa',
               '#cdebe9', '#fbd8e9', '#e4e7d7', '#d0dcfa', '#fbe0c7']
    order = [focus] + sorted((i for i in range(len(neighbors)) if i != focus),
                            key=lambda i: -len(neighbors[i]))
    colors = [-1] * len(neighbors)
    for i in order:
        taken = {colors[j] for j in neighbors[i] if colors[j] >= 0}
        colors[i] = next((k for k in range(len(palette)) if k not in taken),
                         i % len(palette))
    return [palette[i] for i in colors]


class VoronoiSimulation:
    """Geometry + visualization, with reusable snapshots and animated exports."""

    def __init__(self, points: np.ndarray, domain: Domain,
                 focus: int | None = None, circle_radius: float | None = None):
        self.domain = domain
        self.points = np.asarray(points, dtype=float)
        self.cells = compute_cells(self.points, domain)
        self.neighbors = cell_neighbors(self.points, self.cells, domain)
        self.focus = find_focus(self.points, domain, self.neighbors) if focus is None else focus
        if not 0 <= self.focus < len(self.points):
            raise ValueError('focus index must identify a point in the dataset')
        self.local_neighbors = ordered_neighbors(self.points, self.focus, self.neighbors)
        self.reveal_order = breadth_first_order(self.neighbors, self.focus)
        self.colors = _select_colors(self.neighbors, self.focus)
        self.peer = None
        if len(self.points) > 1:
            candidates = np.linalg.norm(self.points - self.points[self.focus], axis=1)
            candidates[self.focus] = np.inf
            self.peer = int(np.argmin(candidates))
            nearest = np.partition(np.linalg.norm(self.points[:, None] - self.points[None, :],
                                                 axis=2) + np.eye(len(self.points)) * 1e8,
                                             0, axis=1)[:, 0]
            median_nn = float(np.median(nearest))
            default_radius = max(0.9 * median_nn,
                                 0.80 * float(candidates[self.peer]),
                                 0.09 * min(domain.width, domain.height))
        else:
            default_radius = 0.17 * min(domain.width, domain.height)
        self.circle_radius = default_radius if circle_radius is None else float(circle_radius)
        if self.circle_radius <= 0:
            raise ValueError('circle radius must be positive')
        self.stats = partition_checks(self.points, self.cells, domain)
        self._cum_seconds = np.cumsum([0] + [p.seconds for p in PHASES])
        self.total_seconds = float(self._cum_seconds[-1])

    def phase_at(self, seconds: float) -> tuple[Phase, float, int]:
        t = float(np.clip(seconds, 0, self.total_seconds - 1e-9))
        idx = int(np.searchsorted(self._cum_seconds, t, side='right') - 1)
        phase = PHASES[idx]
        local = (t - self._cum_seconds[idx]) / phase.seconds
        return phase, float(local), idx

    def time_of(self, phase: str, progress: float = 0.97) -> float:
        idx = next(i for i, p in enumerate(PHASES) if p.key == phase)
        return float(self._cum_seconds[idx] + progress * PHASES[idx].seconds)

    def _figure(self, dpi: int = 110):
        fig = plt.figure(figsize=(12.8, 7.2), dpi=dpi, facecolor=BACKGROUND)
        ax = fig.add_axes([0.055, 0.135, 0.668, 0.755])
        side = fig.add_axes([0.758, 0.13, 0.216, 0.756])
        title = fig.text(0.055, 0.957, '', va='top', color=INK,
                         fontsize=20, fontweight='bold', fontfamily='DejaVu Sans')
        eyebrow = fig.text(0.975, 0.961, 'VORONOI  /  FROM POINTS TO REGIONS',
                           va='top', ha='right', color=MUTED, fontsize=9, fontweight='bold')
        progress_ax = fig.add_axes([0.055, 0.065, 0.92, 0.027])
        progress_ax.set_xlim(0, self.total_seconds)
        progress_ax.set_ylim(0, 1)
        progress_ax.set_axis_off()
        progress_ax.plot([0, self.total_seconds], [0.5, 0.5], lw=3.2,
                         color=GRID, solid_capstyle='round')
        progress_line, = progress_ax.plot([0, 0], [0.5, 0.5], lw=3.2,
                                          color=BLUE, solid_capstyle='round')
        progress_dot, = progress_ax.plot([0], [0.5], marker='o', markersize=6,
                                         color=BLUE)
        return fig, ax, side, title, progress_line, progress_dot

    def _setup_plot(self, ax):
        ax.clear()
        ax.set_facecolor(PLOT_BG)
        ax.set_xlim(0, self.domain.width)
        ax.set_ylim(0, self.domain.height)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#d7e0ea')
            spine.set_linewidth(1.4)

    def _draw_sites(self, ax, focus_on: bool = False):
        ax.scatter(self.points[:, 0], self.points[:, 1], s=32, c=INK,
                   edgecolors=PLOT_BG, linewidth=0.9, zorder=10)
        if focus_on:
            x, y = self.points[self.focus]
            ax.add_patch(Circle((x, y), 0.22 * min(self.domain.width / 10,
                                                self.domain.height / 7.5),
                                fc=BLUE, ec='none', alpha=0.10, zorder=11))
            ax.scatter([x], [y], s=92, c=BLUE, edgecolors='white',
                       linewidths=1.7, zorder=13)

    def _draw_circles(self, ax, alpha: float, radius: float):
        for x, y in self.points:
            ax.add_patch(Circle((x, y), radius=radius, linewidth=1.25,
                                edgecolor=BLUE, facecolor='#78b5f4',
                                alpha=alpha, zorder=2))

    def _draw_overlap(self, ax, alpha: float):
        if self.peer is None:
            return
        a = self.points[self.focus]
        b = self.points[self.peer]
        lens = circle_overlap_polygon(a, b, self.circle_radius)
        if len(lens):
            ax.add_patch(Polygon(lens, closed=True, facecolor=RED, linewidth=1.4,
                                 edgecolor=RED, alpha=0.22 + 0.46 * alpha, zorder=5))
            midpoint = (a + b) / 2
            ax.scatter([midpoint[0]], [midpoint[1]], s=77, marker='x',
                       c=RED, linewidths=2.7, alpha=alpha, zorder=9)

    def _draw_links(self, ax, fraction: float, show_cut_lines: bool = False):
        n = len(self.local_neighbors)
        amount = n * np.clip(fraction, 0, 1)
        a = self.points[self.focus]
        for k, j in enumerate(self.local_neighbors):
            u = float(np.clip(amount - k, 0, 1))
            if u <= 0:
                continue
            b = self.points[j]
            end = _mix(a, b, _ease(u))
            ax.plot([a[0], end[0]], [a[1], end[1]], '-', color='#8595a7',
                    linewidth=1.4 if not show_cut_lines else 1.0,
                    alpha=0.75 if not show_cut_lines else 0.52, zorder=4)

    def _draw_polygon(self, ax, polygon, fill, edge=BLUE,
                      fill_alpha=0.75, lw=2.4, zorder=6):
        if len(polygon) >= 3:
            ax.add_patch(Polygon(polygon, closed=True, facecolor=fill,
                                 edgecolor=edge, linewidth=lw,
                                 alpha=fill_alpha, zorder=zorder, joinstyle='round'))

    def _draw_cell_construction(self, ax, progress: float):
        n = len(self.local_neighbors)
        if n == 0:
            self._draw_polygon(ax, self.cells[self.focus], '#b8dff9')
            return 0, 0
        amount = min(n, n * float(np.clip(progress, 0, 1)))
        completed = min(n, int(amount + 1e-9))
        partial = amount - completed
        active = self.local_neighbors[completed] if completed < n else None
        polygon = self.domain.polygon()
        for j in self.local_neighbors[:completed]:
            polygon = clip_closer_to_a(polygon, self.points[self.focus], self.points[j])
        self._draw_polygon(ax, polygon, '#b8dff9', edge=BLUE,
                           fill_alpha=0.28 if completed < 2 else 0.72, lw=2.1)
        a = self.points[self.focus]
        for j in self.local_neighbors[:completed]:
            segment = bisector_segment(a, self.points[j], self.domain)
            if len(segment) == 2:
                ax.plot(segment[:, 0], segment[:, 1], linestyle=(0, (4, 4)),
                        color=VIOLET, alpha=0.29, linewidth=1.3, zorder=7)
        if active is not None:
            b = self.points[active]
            segment = bisector_segment(a, b, self.domain)
            midpoint = (a + b) / 2
            # Reveal the bisector outward from the exact midpoint, smoothly.
            if len(segment) == 2:
                u = _ease(partial)
                short = np.vstack([_mix(midpoint, segment[0], u),
                                   _mix(midpoint, segment[1], u)])
                ax.plot(short[:, 0], short[:, 1], linestyle=(0, (5, 3)),
                        color=VIOLET, linewidth=2.55, zorder=8)
            ax.scatter([midpoint[0]], [midpoint[1]], s=49, marker='s',
                       c=VIOLET, edgecolors='white', linewidths=0.7, zorder=9)
            ax.scatter([b[0]], [b[1]], s=85, c=AMBER, edgecolors='white',
                       linewidths=1.4, zorder=12)
        return completed, n

    def _draw_tessellation(self, ax, fraction: float, final: bool = False):
        n = len(self.reveal_order)
        amount = n if final else min(n, float(np.clip(fraction, 0, 1)) * n)
        for k, i in enumerate(self.reveal_order):
            opacity = float(np.clip(amount - k, 0, 1))
            if opacity < 1e-5:
                continue
            self._draw_polygon(ax, self.cells[i], self.colors[i],
                               edge='#344d66' if final else '#617993',
                               fill_alpha=0.97 * opacity,
                               lw=1.35 if final else 1.25, zorder=3)
        self._draw_polygon(ax, self.cells[self.focus], self.colors[self.focus],
                           edge=BLUE, fill_alpha=0.22 if final else 0.12,
                           lw=2.5, zorder=4)

    def _draw_side(self, side, phase: Phase, u: float, stage_index: int,
                   complete_constraints: int = 0):
        side.clear()
        side.set_axis_off()
        side.set_xlim(0, 1)
        side.set_ylim(0, 1)
        side.axvline(x=0.035, color='#dbe4ed', lw=1.2, ymin=0.01, ymax=0.99)
        side.text(0.105, 0.949, f'STEP {stage_index + 1:02d} / {len(PHASES):02d}',
                  ha='left', va='top', color=BLUE, fontweight='bold', fontsize=10)
        side.text(0.105, 0.855, phase.detail, fontsize=12.4,
                  color=INK, ha='left', va='top', wrap=True,
                  linespacing=1.65, transform=side.transAxes)
        side.text(0.105, 0.617, phase.note, fontsize=10.3,
                  color=MUTED, ha='left', va='top', wrap=True,
                  linespacing=1.55, transform=side.transAxes)

        if phase.key in ('focus', 'links', 'cuts', 'cell', 'fill', 'done'):
            y = 0.42
            side.text(0.105, y, 'DISTANCE RULE', fontsize=9.5, color=MUTED,
                      fontweight='bold', va='top')
            side.text(0.105, y - 0.075, r'$\|x-p_i\|^2\leq\|x-p_j\|^2$',
                      fontsize=14.2, color=INK, va='top')
            side.text(0.105, y - 0.175, 'Keep the half-plane\ncloser to site i.',
                      fontsize=10.4, color=MUTED, va='top', linespacing=1.45)

        if phase.key == 'cuts':
            side.text(0.105, 0.105,
                      f'Half-planes: {complete_constraints} / {len(self.local_neighbors)}',
                      color=DEEP_BLUE, fontsize=11, fontweight='bold')
        elif phase.key in ('links', 'cell'):
            side.text(0.105, 0.105, f'Voronoi neighbors: {len(self.local_neighbors)}',
                      color=DEEP_BLUE, fontsize=11, fontweight='bold')
        elif phase.key in ('fill', 'done'):
            count = len(self.points) if phase.key == 'done' else max(1, int(np.ceil(u * len(self.points))))
            side.text(0.105, 0.105, f'Regions: {count} / {len(self.points)}',
                      color=DEEP_BLUE, fontsize=11, fontweight='bold')

    def draw_frame(self, t: float, parts):
        fig, ax, side, title, progress_line, progress_dot = parts
        phase, u, index = self.phase_at(t)
        title.set_text(phase.heading)
        self._setup_plot(ax)
        active_point = phase.key not in ('points', 'circles', 'overlap', 'clear')
        constraints_completed = 0

        if phase.key == 'circles':
            self._draw_circles(ax, alpha=0.19, radius=self.circle_radius * _ease(u))
        elif phase.key == 'overlap':
            self._draw_circles(ax, alpha=0.19, radius=self.circle_radius)
            self._draw_overlap(ax, alpha=_ease(u))
        elif phase.key == 'clear':
            fade = 1.0 - _ease(u)
            self._draw_circles(ax, alpha=0.19 * fade, radius=self.circle_radius)
            if self.peer is not None and fade > 0.03:
                self._draw_overlap(ax, alpha=fade)
        elif phase.key == 'focus':
            pass
        elif phase.key == 'links':
            self._draw_links(ax, fraction=u)
        elif phase.key == 'cuts':
            self._draw_links(ax, fraction=1, show_cut_lines=True)
            constraints_completed, _ = self._draw_cell_construction(ax, progress=u)
        elif phase.key == 'cell':
            self._draw_polygon(ax, self.cells[self.focus], '#b8dff9',
                               edge=BLUE, fill_alpha=0.88)
        elif phase.key in ('fill', 'done'):
            self._draw_tessellation(ax, fraction=_ease(u), final=phase.key == 'done')

        self._draw_sites(ax, focus_on=active_point)
        self._draw_side(side, phase, u, index, constraints_completed)
        progress_line.set_data([0, min(t, self.total_seconds)], [0.5, 0.5])
        progress_dot.set_data([min(t, self.total_seconds)], [0.5])
        return []

    def save_animation(self, path: str | Path, fps: int = 12,
                       dpi: int = 100, speed: float = 1.0):
        if fps < 1 or speed <= 0:
            raise ValueError('fps must be >= 1; speed must be positive')
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() not in ('.mp4', '.gif', '.webm'):
            raise ValueError('For animation, choose an output .mp4, .webm or .gif')
        frames = max(2, int(np.ceil(self.total_seconds / speed * fps)))
        parts = self._figure(dpi=dpi)
        fig = parts[0]
        def update(frame_no):
            t = frame_no * speed / fps
            return self.draw_frame(t, parts)
        movie = animation.FuncAnimation(fig, update, frames=frames,
                                        interval=1000 / fps, repeat=False,
                                        blit=False)
        writer = (animation.PillowWriter(fps=fps)
                  if path.suffix.lower() == '.gif'
                  else animation.FFMpegWriter(fps=fps, codec='libx264'
                                              if path.suffix.lower() == '.mp4' else 'libvpx-vp9',
                                              bitrate=1900, extra_args=['-pix_fmt', 'yuv420p']
                                              if path.suffix.lower() == '.mp4' else []))
        try:
            movie.save(str(path), writer=writer, dpi=dpi)
        finally:
            plt.close(fig)

    def save_snapshot(self, path: str | Path, phase: str = 'done', dpi: int = 140):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        parts = self._figure(dpi=dpi)
        try:
            self.draw_frame(self.time_of(phase), parts)
            parts[0].savefig(path, dpi=dpi, facecolor=BACKGROUND)
        finally:
            plt.close(parts[0])

    def save_storyboard(self, path: str | Path, dpi: int = 105):
        """Create a 3x3 storyboard from *real rendered* animation stages."""
        snapshots = [('points', .8), ('circles', .97), ('overlap', .97),
                     ('focus', .97), ('links', .97), ('cuts', .43),
                     ('cell', .97), ('fill', .57), ('done', .97)]
        frames = []
        for key, fraction in snapshots:
            parts = self._figure(dpi=dpi)
            fig = parts[0]
            self.draw_frame(self.time_of(key, fraction), parts)
            fig.canvas.draw()
            frame = Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert('RGB')
            frames.append(frame)
            plt.close(fig)
        thumb_size = (960, 540)
        margin = 15
        width, height = 3 * thumb_size[0] + 4 * margin, 3 * thumb_size[1] + 4 * margin
        result = Image.new('RGB', (width, height), '#e7edf4')
        for i, frame in enumerate(frames):
            frame.thumbnail(thumb_size, Image.Resampling.LANCZOS)
            x = margin + (i % 3) * (thumb_size[0] + margin)
            y = margin + (i // 3) * (thumb_size[1] + margin)
            result.paste(frame, (x, y))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        result.save(path)
