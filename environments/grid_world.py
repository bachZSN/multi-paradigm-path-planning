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
        self.grid = np.zeros((self.width, self.height))
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
        return self.grid[x, y] >= 0 and self.grid[x, y] != float('inf')

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

    def add_obstacle(self, x, y, height=1.0):
        self.grid[x, y] = height
        self.update_heights()

    def add_mountain(self, x, y, height=1.0, radius=5, function="relu"):
        """
        Add a mountain with a summit at (x, y) and gradually decreasing height.

        Args:
            x (int): X-coordinate of the summit.
            y (int): Y-coordinate of the summit.
            height (float): Maximum height at the summit.
            radius (int): Radius of the mountain's influence.
            function (str): The function to use for height decay ("relu" or "arctan").
        """
        for i in range(self.height):
            for j in range(self.width):
                # Calculate the distance from the summit
                distance = np.sqrt((x - j) ** 2 + (y - i) ** 2)
                if distance <= radius and self.is_valid((i, j)):
                    self.grid[i, j] += height * (1 - np.arctan(distance) / np.pi)

                # Calculate the height based on the chosen function
                    if function == "relu":
                        self.grid[i, j] += max(0, height * (1 - distance / radius))
                    elif function == "arctan":
                        self.grid[i, j] += height * (1 - np.arctan(distance) / np.pi)
        self.update_heights()

def create_default_world():
    world = GridWorld(100)

    # add some obstacles
    for i in range(20):
        [x,y] = np.random.randint(0, 100, size=2)
        world.add_mountain(x, y, height=10.0, radius=20)
    return world

class Agent:
    def __init__(self, start, goal):
        self.start = start
        self.goal = goal
