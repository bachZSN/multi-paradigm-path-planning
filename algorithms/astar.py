import heapq
import queue
import numpy as np

def bfs(start, goal, grid):
    """Breadth-First Search algorithm for pathfinding on a grid.
    Args:
        start (tuple): Starting position (x, y).
        goal (tuple): Goal position (x, y).
        grid (GridWorld): the grid world containing terrain and validity checks.
    Returns:
        tuple: (came_from, path) where came_from is the parent map and path is the reconstructed path.
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

def dijkstra_search(start, goal, grid):
    """Dijkstra's algorithm for pathfinding on a grid.
    Args:
        start (tuple): Starting position (x, y).
        goal (tuple): Goal position (x, y).
        grid (GridWorld): the grid world containing terrain and validity checks.
    Returns:
        tuple: (came_from, path) where came_from is the parent map and path is the reconstructed path.
    """
    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}
    cost_move_dir = 1  # Base cost to move to a neighbor (can be adjusted based on terrain)

    while frontier:
        cost, current = heapq.heappop(frontier)

        if current == goal:
            break

        for next in grid.neighbors(current):
            new_cost = cost_so_far[current] + cost_move_dir + abs(grid.grid[next[1],next[0]]-grid.grid[current[1],current[0]])  # Cost is the height difference
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
        grid (GridWorld): the grid world containing terrain and validity checks.
    Returns:
        tuple: (came_from, path) where came_from is the parent map and path is the reconstructed path.
    """
    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}
    cost_move_dir = 1  # Base cost to move to a neighbor (can be adjusted based on terrain)

    while frontier:
        cost , current = heapq.heappop(frontier)

        if current == goal:
            break

        for next in grid.neighbors(current):
            new_cost = cost_so_far[current] + cost_move_dir + abs(grid.grid[next[1],next[0]]-grid.grid[current[1],current[0]])  # Cost is the height difference
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(next, goal, grid)
                heapq.heappush(frontier, (priority, next))
                came_from[next] = current

    return came_from, reconstruct_path(came_from, start, goal)

def heuristic(a, b, grid):
    """Manhattan distance heuristic (admissible for 4-directional movement)."""
    (a1, b1) = a
    (a2, b2) = b
    return abs(a1 - a2) + abs(b1 - b2)

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

def calculate_total_cost(path, grid):
    """Calculate the total cost of a path (movement cost + height differences).
    Args:
        path (list): A list of (x, y) positions along the path.
        grid (GridWorld): the grid world containing terrain heights.
    Returns:
        float: Total traversal cost.
    """
    if len(path) < 2:
        return 0  # No cost for a path with fewer than 2 points

    cost_move_dir = 1  # Base cost to move to a neighbor (can be adjusted based on terrain)

    # Use sum with a generator expression for consecutive pairs
    total_cost = sum(
        cost_move_dir + abs(grid.grid[next_pos[1], next_pos[0]] - grid.grid[current[1], current[0]])
        for current, next_pos in zip(path, path[1:])
    )

    return total_cost
