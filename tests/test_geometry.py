import numpy as np
import pytest
from voronoi_lab.geometry import (
    Domain, uniform_sites, compute_cells, clip_closer_to_a, bisector_halfplane,
    polygon_area, partition_checks, cell_neighbors, ordered_neighbors,
)


def test_uniform_points_are_reproducible_and_bounded():
    a = uniform_sites(100, Domain(10, 8), seed=109)
    b = uniform_sites(100, Domain(10, 8), seed=109)
    assert np.array_equal(a, b)
    assert a.shape == (100, 2)
    assert np.all(a > 0) and np.all(a < [10, 8])


def test_bisector_is_perpendicular_at_midpoint():
    a = np.array([1.0, 2.0])
    b = np.array([4.0, 6.0])
    n, t = bisector_halfplane(a, b)
    assert n @ ((a + b) / 2) == pytest.approx(t)
    assert n @ np.array([-4.0, 3.0]) == pytest.approx(0)


def test_two_site_split_vertical():
    domain = Domain(10, 8)
    sites = np.array([[3.0, 4.0], [7.0, 4.0]])
    cells = compute_cells(sites, domain)
    assert polygon_area(cells[0]) == pytest.approx(40)
    assert polygon_area(cells[1]) == pytest.approx(40)
    assert max(cells[0][:, 0]) == pytest.approx(5)
    assert min(cells[1][:, 0]) == pytest.approx(5)
    assert cell_neighbors(sites, cells, domain) == [[1], [0]]


def test_clipping_keeps_only_nearer_side():
    domain = Domain(10, 10)
    a, b = np.array([2, 5]), np.array([8, 5])
    output = clip_closer_to_a(domain.polygon(), a, b)
    assert output.shape == (4, 2)
    assert np.max(output[:, 0]) == pytest.approx(5.0)


@pytest.mark.parametrize("seed,n", [(1, 2), (2, 5), (12, 15), (91, 28), (333, 60)])
def test_random_cells_cover_domain_without_distance_violations(seed, n):
    domain = Domain(10, 8)
    sites = uniform_sites(n, domain, seed=seed)
    cells = compute_cells(sites, domain)
    stats = partition_checks(sites, cells, domain)
    assert stats["area_error"] < 1e-7
    assert stats["maximum_constraint_violation"] < 1e-7
    assert all(polygon_area(cell) > 0 for cell in cells)
    neighbors = cell_neighbors(sites, cells, domain)
    for i, adjacent in enumerate(neighbors):
        assert i not in adjacent
        assert all(i in neighbors[j] for j in adjacent)
        assert all(j != i for j in ordered_neighbors(sites, i, neighbors))


def test_single_site_owns_whole_domain():
    domain = Domain(11, 7)
    sites = np.array([[5.0, 4.0]])
    assert polygon_area(compute_cells(sites, domain)[0]) == pytest.approx(77)
    assert cell_neighbors(sites, compute_cells(sites, domain), domain) == [[]]


def test_invalid_sites_rejected():
    domain = Domain()
    with pytest.raises(ValueError, match="Duplicate"):
        compute_cells(np.array([[2, 3], [2, 3]]), domain)
    with pytest.raises(ValueError, match="within"):
        compute_cells(np.array([[11, 2]]), domain)


def test_random_probe_locations_belong_to_nearest_site_cell():
    domain = Domain(10, 8)
    sites = uniform_sites(26, domain, seed=16)
    cells = compute_cells(sites, domain)
    rng = np.random.default_rng(141)
    probes = rng.uniform([0, 0], [domain.width, domain.height], size=(400, 2))

    def inside_convex_ccw_polygon(q, polygon):
        for i, end in enumerate(polygon):
            begin = polygon[i-1]
            side = (end[0] - begin[0]) * (q[1] - begin[1]) - (end[1] - begin[1]) * (q[0] - begin[0])
            if side < -1e-8:
                return False
        return True

    for q in probes:
        winner = int(np.argmin(np.sum((sites - q)**2, axis=1)))
        assert inside_convex_ccw_polygon(q, cells[winner])

    for polygon in cells:
        assert np.min(polygon) >= -1e-9
        assert np.max(polygon[:, 0]) <= domain.width + 1e-9
        assert np.max(polygon[:, 1]) <= domain.height + 1e-9
