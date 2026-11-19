"""Edit and run this file to experiment with your own Voronoi algorithm.

Run:
    python workbench.py

Start by changing COUNT, SEED, FOCUS and DOMAIN. The actual implementation
lives in voronoi_lab/geometry.py; change compute_cells() there to replace
this algorithm, and the animation will automatically show the new result.
"""

from voronoi_lab.geometry import Domain, uniform_sites, polygon_area
from voronoi_lab.visuals import VoronoiSimulation

# ---------- Editable experiment parameters ----------
COUNT = 18
SEED = 42
DOMAIN = Domain(width=10, height=7.5)
FOCUS = None                 # None: automatically choose a near-central site
CIRCLE_RADIUS = None         # None: choose radius that exposes overlaps
FPS = 12
# ----------------------------------------------------

points = uniform_sites(COUNT, DOMAIN, seed=SEED)
sim = VoronoiSimulation(points, DOMAIN, focus=FOCUS,
                        circle_radius=CIRCLE_RADIUS)

print('Focus site:', sim.focus)
print('Neighbors:', sim.neighbors[sim.focus])
print('Focus cell vertices:')
print(sim.cells[sim.focus])
print('Focus cell area:', polygon_area(sim.cells[sim.focus]))
print('Partition check:', sim.stats)

sim.save_animation('my_voronoi.mp4', fps=FPS, dpi=105)
sim.save_snapshot('my_voronoi.svg')
sim.save_storyboard('my_storyboard.png')
print('Saved my_voronoi.mp4, my_voronoi.svg and my_storyboard.png')
