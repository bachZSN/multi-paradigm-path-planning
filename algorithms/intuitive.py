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
    #as for the decision on when to look ahead when to follow the path with lower height difference when to pause and try to calculate a better heuristic I want to train a model that can learn from experience when to do what, but for now we will just use a simple rule based approach where we look ahead if we are at a certain distance from the goal or if we encounter a certain height difference
