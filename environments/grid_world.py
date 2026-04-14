"""
Grid world environment used across all planning paradigms.

This module defines a unified representation of the environment,
including obstacles and validity checks, so that all methods
(A*, CSP, gradient-based optimization) operate on the same problem instance.
"""
class GridWorld:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.obstacles = set()

    def add_obstacle(self, x, y):
        self.obstacles.add((x, y))

    def is_valid(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height and (x, y) not in self.obstacles

def create_default_world():
    world = GridWorld(20, 20)

    # add some obstacles
    for i in range(1, 19):
        world.add_obstacle(10, i)
        world.add_obstacle(i, 10)

    return world

class Agent:
    def __init__(self, start, goal):
        self.start = start
        self.goal = goal
