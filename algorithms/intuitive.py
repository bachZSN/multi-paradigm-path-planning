import numpy as np
import heapq
import torch

def intuitive(self, start, goal, grid):
    """Intuitive search algorithm that prioritizes moving towards the goal while considering height differences. This algorithm is not guaranteed to find the optimal path, but it can be faster in certain scenarios."""

    #we will simulate a human walking through terrain by always trying to move towards the goal, but also considering the height differences. We will use a priority queue to explore nodes that are closer to the goal and have lower height differences first.
    #we also only use a rough estimate as a heuristic at any given time
    #only after a while or for difficult decisions we will try to look ahead and consider the terrain more carefully using a better heuristic
    #generally we tend to prefer paths that point to the goal, but we also want to avoid steep climbs if possible
    #but we have some sort of limited vision and we can only see a certain radius around us, so we will use a heuristic that considers the height differences in that radius and the direction towards the goal
    #certain height differences might be more acceptable if they are in the direction of the goal, while others might be less acceptable if they are in the opposite direction
    #for a different approach we can simlulate human instinct to climb a higher place to ascertain the terrain, then deciding on a path

    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}

    while frontier:
        cost, current = heapq.heappop(frontier)

        if current == goal:
            break

        for next in grid.neighbors(current):
            # Calculate the intuitive cost based on direction and height difference
            direction_cost = self.direction_cost(current, next, goal)
            height_cost = abs(grid.grid[next] - grid.grid[current])
            new_cost = cost_so_far[current] + direction_cost + height_cost

            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost
                heapq.heappush(frontier, (priority, next))
                came_from[next] = current

    return came_from, self.reconstruct_path(came_from, start, goal)
