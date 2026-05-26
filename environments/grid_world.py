from typing import Iterator
import numpy as np
GridLocation = tuple[int, int]

"""
Grid world environment used across all planning paradigms.

This module defines a unified representation of the environment,
including obstacles and validity checks, so that all methods
(A*, CSP, gradient-based optimization) operate on the same problem instance.
"""
class GridWorld:
    def __init__(self, dimension: int):
        self.width = dimension
        self.height = dimension
        self.grid = np.zeros((self.height, self.width), dtype=np.float64)
        self.max_height = 0
        self.min_height = 0
        self.update_heights()  # Initialize the cached values

    def is_valid(self, id: GridLocation) -> bool:
        return self.in_bounds(id) and self.passable(id)

    def in_bounds(self, id: GridLocation) -> bool:
        x, y = id
        return 0 <= x < self.width and 0 <= y < self.height

    def passable(self, id: GridLocation) -> bool:
        x, y = id
        val = self.grid[y, x]
        return val >= 0 and not np.isinf(val)

    def neighbors(self, id: GridLocation) -> Iterator[GridLocation]:
        x, y = id
        neighbors = [(x + dx, y + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]]
        valid_neighbors = filter(self.is_valid, neighbors)
        return valid_neighbors

    def update_heights(self):
        """Update the cached max and min heights."""
        valid_heights = self.grid[(self.grid != -1) & (self.grid != float('inf'))]
        self.max_height = valid_heights.max() if valid_heights.size > 0 else 0
        self.min_height = valid_heights.min() if valid_heights.size > 0 else 0

    def add_obstacle(self, x, y, height=1):
        self.grid[y, x] = height
        self.update_heights()

    def add_mountain(self, x, y, height=1, radius=5, function="relu"):
        # Vectorized implementation (much faster than Python loops).
        yy, xx = np.indices((self.height, self.width))
        dist = np.sqrt((x - xx) ** 2 + (y - yy) ** 2)
        mask = dist <= radius

        # Keep semantics of is_valid/passable for impassable cells.
        passable = (self.grid >= 0) & (~np.isinf(self.grid))
        mask &= passable

        if not np.any(mask):
            return

        if function == "relu":
            delta = height * (1 - dist / radius)
            delta = np.maximum(delta, 0.0)
        elif function == "arctan":
            delta = height * (1 - np.arctan(dist) / np.pi)
        else:
            raise ValueError(f"Unknown mountain function: {function}")

        self.grid[mask] += delta[mask]
        self.update_heights()

def create_default_world():
    default_size = 200
    world = GridWorld(default_size)

    # add some obstacles
    for i in range(50):
        [x,y] = np.random.randint(0, default_size, size=2)
        world.add_mountain(x, y, height=10.0, radius=20)
    return world

class Agent:
    def __init__(self, start, goal):
        self.start = start
        self.goal = goal
