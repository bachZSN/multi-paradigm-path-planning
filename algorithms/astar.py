import heapq
import queue
import numpy as np

def bfs(start, goal, grid):
    """Breadth-First Search algorithm for pathfinding on a grid.
    Args:
        start (tuple): Starting position (x, y).
        goal (tuple): Goal position (x, y).
        grid (np.ndarray): 2D grid representing the heights of the terrain.
    Returns:
        dict: A dictionary mapping each visited node to its parent node, which can be used to reconstruct the path.
    """
    frontier = queue.Queue()
    frontier.put(start)
    came_from = {start: None}

    while not frontier.empty():
        current = frontier.get()

        if current == goal:
            break

        for next in grid.neighbors(current):
            if next not in came_from:
                frontier.put(next)
                came_from[next] = current

    return came_from, reconstruct_path(came_from, start, goal)

def dijsktra_search(start, goal, grid):
    """Dijkstra's algorithm for pathfinding on a grid.
    Args:
        start (tuple): Starting position (x, y).
        goal (tuple): Goal position (x, y).
        grid (np.ndarray): 2D grid representing the heights of the terrain.
    Returns:
        dict: A dictionary mapping each visited node to its parent node, which can be used to reconstruct the path.
    """
    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}

    while frontier:
        cost, current = heapq.heappop(frontier)

        if current == goal:
            break

        for next in grid.neighbors(current):
            new_cost = cost_so_far[current] + abs(grid.grid[next]-grid.grid[current])  # Cost is the height difference
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost
                heapq.heappush(frontier, (priority, next))
                came_from[next] = current

    return came_from, reconstruct_path(came_from, start, goal)

def astar(start, goal, grid):
    """A* search algorithm for pathfinding on a grid.
    Args:
        start (tuple): Starting position (x, y).
        goal (tuple): Goal position (x, y).
        grid (np.ndarray): 2D grid representing the heights of the terrain.
    Returns:
        dict: A dictionary mapping each visited node to its parent node, which can be used to reconstruct the path.
    """
    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}

    while frontier:
        cost , current = heapq.heappop(frontier)

        if current == goal:
            break

        for next in grid.neighbors(current):
            new_cost = cost_so_far[current] + abs(grid.grid[next]-grid.grid[current])  # Cost is the height difference
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(next, goal, grid)
                heapq.heappush(frontier, (priority, next))
                came_from[next] = current



    return came_from, reconstruct_path(came_from, start, goal)

def heuristic(a, b, grid):
    """Heuristic function for A* search, using Manhattan distance.
    Args:
        a (tuple): Position (x, y) of the first point.
        b (tuple): Position (x, y) of the second point.
    Returns:
        int: The Manhattan distance between points a and b plus the absolute height difference between the two points.
    """
    (a1, b1) = a
    (a2, b2) = b
    return abs(a1 - a2) + abs(b1 - b2) + abs(grid.grid[a] - grid.grid[b])

def reconstruct_path(came_from, start, goal):
    path = []
    current = goal
    if current not in came_from:
        return []
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    return path
