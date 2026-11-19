"""Small, dependency-light implementation of rectangularly clipped Voronoi cells.

Only NumPy is required.  The core operation is Sutherland-Hodgman clipping
against the perpendicular-bisector half-plane for a pair of sites.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import deque

import numpy as np


@dataclass(frozen=True)
class Domain:
    width: float = 10.0
    height: float = 7.5

    def __post_init__(self):
        if not (self.width > 0 and self.height > 0):
            raise ValueError("Domain width and height must be positive")

    def polygon(self) -> np.ndarray:
        return np.array([[0.0, 0.0], [self.width, 0.0],
                         [self.width, self.height], [0.0, self.height]], dtype=float)


def uniform_sites(n: int, domain: Domain, seed: int = 42,
                  margin: float = 0.07) -> np.ndarray:
    """Sample independent x and y coordinates uniformly (not normally).

    `margin` is the fraction of the width/height kept empty near boundaries.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    if not (0 <= margin < 0.5):
        raise ValueError("margin must be in [0, 0.5)")
    rng = np.random.default_rng(seed)
    unit_points = rng.uniform(margin, 1.0 - margin, size=(n, 2))
    return unit_points * np.array([domain.width, domain.height])


def validate_sites(sites: np.ndarray, domain: Domain) -> np.ndarray:
    sites = np.asarray(sites, dtype=float)
    if sites.ndim != 2 or sites.shape[1] != 2 or len(sites) < 1:
        raise ValueError("sites must be a nonempty Nx2 array")
    if not np.isfinite(sites).all():
        raise ValueError("All site coordinates must be finite")
    if (sites < 0).any() or (sites[:, 0] > domain.width).any() or (sites[:, 1] > domain.height).any():
        raise ValueError("Every site must be within [0,width] x [0,height]")
    if len(np.unique(sites, axis=0)) < len(sites):
        raise ValueError("Duplicate site coordinates are not supported")
    return sites


def bisector_halfplane(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    """Return n, c defining the points closer to a than b: n dot x <= c.

    Expanding squared Euclidean distances:
       ||x-a||^2 <= ||x-b||^2
          (b-a) dot x <= (||b||^2 - ||a||^2)/2
    The bounding line passes through (a+b)/2 and is perpendicular to b-a.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return b - a, 0.5 * (float(b @ b) - float(a @ a))


def _dedupe_neighbors(vertices: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    if len(vertices) <= 1:
        return vertices
    output = [vertices[0]]
    for vertex in vertices[1:]:
        if np.linalg.norm(vertex - output[-1]) > epsilon:
            output.append(vertex)
    if len(output) > 1 and np.linalg.norm(output[-1] - output[0]) <= epsilon:
        output.pop()
    return np.asarray(output, dtype=float).reshape((-1, 2))


def clip_to_halfplane(polygon: np.ndarray, normal: np.ndarray,
                      threshold: float, epsilon: float = 1e-11) -> np.ndarray:
    """Clip a convex polygon to normal dot x <= threshold.

    This is Sutherland-Hodgman clipping, O(number of polygon edges).
    Polygons have counterclockwise vertices; the result keeps that ordering.
    """
    polygon = np.asarray(polygon, dtype=float)
    if not len(polygon):
        return np.empty((0, 2), dtype=float)

    # Evaluate signed distance from the infinite clipping line (unscaled).
    f = polygon @ normal - threshold
    result = []
    for k in range(len(polygon)):
        start = polygon[k - 1]
        end = polygon[k]
        start_f, end_f = f[k - 1], f[k]
        start_inside = start_f <= epsilon
        end_inside = end_f <= epsilon
        if start_inside != end_inside:
            denom = start_f - end_f
            if abs(denom) > 1e-30:
                t = np.clip(start_f / denom, 0.0, 1.0)
                result.append(start + t * (end - start))
        if end_inside:
            result.append(end)
    if not result:
        return np.empty((0, 2), dtype=float)
    return _dedupe_neighbors(np.array(result, dtype=float))


def clip_closer_to_a(polygon: np.ndarray, a: np.ndarray,
                     b: np.ndarray) -> np.ndarray:
    normal, threshold = bisector_halfplane(a, b)
    return clip_to_halfplane(polygon, normal, threshold)


def compute_cells(sites: np.ndarray, domain: Domain) -> list[np.ndarray]:
    """Voronoi cells via *all* competing point half-planes; no SciPy needed.

    The union of the cells is the rectangular domain, with disjoint interiors.
    This is O(N^2) clipping; ideal for demonstrating/implementing the method,
    rather than very large point clouds (where Fortune's algorithm is faster).
    """
    sites = validate_sites(sites, domain)
    cells = []
    for i, a in enumerate(sites):
        polygon = domain.polygon()
        for j, b in enumerate(sites):
            if i != j:
                polygon = clip_closer_to_a(polygon, a, b)
                if len(polygon) == 0:
                    raise ArithmeticError("A cell became empty; check input degeneracy")
        cells.append(polygon)
    return cells


def polygon_area(polygon: np.ndarray) -> float:
    if len(polygon) < 3:
        return 0.0
    x, y = polygon[:, 0], polygon[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _on_box_boundary(a: np.ndarray, b: np.ndarray,
                     domain: Domain, eps: float = 1e-7) -> bool:
    for k, val in [(0, 0.0), (0, domain.width), (1, 0.0), (1, domain.height)]:
        if abs(a[k] - val) < eps and abs(b[k] - val) < eps:
            return True
    return False


def cell_neighbors(sites: np.ndarray, cells: list[np.ndarray],
                   domain: Domain) -> list[list[int]]:
    """Read actual Voronoi neighbors from each cell's non-domain edges.

    At the midpoint of a shared edge, the two corresponding sites have equal
    squared distance. Using these edges excludes 'k-nearest' false neighbors.
    """
    sites = validate_sites(sites, domain)
    neighbors = [set() for _ in sites]
    for i, polygon in enumerate(cells):
        for k in range(len(polygon)):
            a, b = polygon[k - 1], polygon[k]
            if np.linalg.norm(a - b) < 1e-8 or _on_box_boundary(a, b, domain):
                continue
            midpoint = (a + b) * 0.5
            d2 = np.sum((sites - midpoint)**2, axis=1)
            delta = abs(d2 - d2[i])
            delta[i] = np.inf
            j = int(np.argmin(delta))
            if delta[j] < 1e-7 * max(1.0, float(d2[i])):
                neighbors[i].add(j)
                neighbors[j].add(i)
    return [sorted(s) for s in neighbors]


def find_focus(sites: np.ndarray, domain: Domain, neighbors: list[list[int]]) -> int:
    """Choose a near-center site with enough neighboring cells for the lesson."""
    midpoint = np.array([domain.width / 2, domain.height / 2])
    distances = np.linalg.norm((sites - midpoint) / [domain.width, domain.height], axis=1)
    # Most central sites work; avoid one with fewer than 4 neighbors when possible.
    complexity_penalty = np.maximum(0, 4 - np.array([len(n) for n in neighbors])) * 0.12
    return int(np.argmin(distances + complexity_penalty))


def ordered_neighbors(sites: np.ndarray, focus: int,
                      neighbors: list[list[int]]) -> list[int]:
    """Sort Voronoi-adjacent sites by polar angle for orderly line drawing."""
    return sorted(neighbors[focus],
                  key=lambda j: float(np.arctan2(*(sites[j] - sites[focus])[::-1])))


def breadth_first_order(neighbors: list[list[int]], focus: int) -> list[int]:
    """Visit cells in rings around the focus, using Voronoi adjacency."""
    queue = deque([focus])
    visited = {focus}
    order = []
    while queue:
        i = queue.popleft()
        order.append(i)
        for j in neighbors[i]:
            if j not in visited:
                visited.add(j)
                queue.append(j)
    # Every bounded Voronoi diagram of distinct sites is connected, but be safe.
    order.extend(j for j in range(len(neighbors)) if j not in visited)
    return order


def bisector_segment(a: np.ndarray, b: np.ndarray,
                     domain: Domain) -> np.ndarray:
    """Endpoints of the infinite bisector, clipped to the plotting rectangle."""
    normal, threshold = bisector_halfplane(a, b)
    width, height = domain.width, domain.height
    possible = []
    if abs(normal[1]) > 1e-12:
        for x in (0.0, width):
            y = (threshold - normal[0] * x) / normal[1]
            if -1e-8 <= y <= height + 1e-8:
                possible.append(np.array([x, np.clip(y, 0, height)]))
    if abs(normal[0]) > 1e-12:
        for y in (0.0, height):
            x = (threshold - normal[1] * y) / normal[0]
            if -1e-8 <= x <= width + 1e-8:
                possible.append(np.array([np.clip(x, 0, width), y]))
    if len(possible) < 2:
        return np.empty((0, 2), dtype=float)
    max_dist = -1.0
    answer = None
    for i, p in enumerate(possible):
        for q in possible[i + 1:]:
            d = float(np.linalg.norm(p - q))
            if d > max_dist:
                max_dist = d
                answer = np.array([p, q])
    return answer if answer is not None else np.empty((0, 2), dtype=float)


def partition_checks(sites: np.ndarray, cells: list[np.ndarray], domain: Domain) -> dict:
    """Diagnostics used by the tests and available to downstream experiments."""
    total = sum(polygon_area(c) for c in cells)
    rectangular_area = domain.width * domain.height
    max_site_violation = 0.0
    for i, polygon in enumerate(cells):
        if not len(polygon):
            continue
        for j in range(len(sites)):
            if j == i:
                continue
            normal, threshold = bisector_halfplane(sites[i], sites[j])
            max_site_violation = max(max_site_violation,
                                     float(max(0.0, np.max(polygon @ normal - threshold))))
    return {"n_cells": len(cells), "area_sum": total,
            "domain_area": rectangular_area,
            "area_error": abs(total - rectangular_area),
            "maximum_constraint_violation": max_site_violation}
