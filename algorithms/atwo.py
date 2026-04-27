import heapq

class Node:
    def __init__(self, position, parent=None):
        self.postion = position
        self.parent = parent
        self.g = 0
        self.h = 0
        self.f = 0

# Convert the grid world to a 2D grid representation
def grid_world_to_grid(grid_world):
    grid = [[0 for _ in range(grid_world.width)] for _ in range(grid_world.height)]
    for x, y in grid_world.obstacles:
        grid[y][x] = 1  # Mark obstacles as 1
    return grid

# A* algorithm implementation
def astar(start, goal, grid):
    if grid[start[0], start[1]] == grid[goal[0], goal[1]]:
        print("Start is already at goal.")

    open_list = []
    closed_list = set()
    start_node = Node(start)
    goal_node = Node(goal)
    heapq.heappush(open_list, start_node)

    while open_list:
        current_node = heapq.heappop(open_list)
        closed_list.add(current_node.postion)

        if current_node.postion == goal_node.postion:
            return reconstruct_path(current_node)

        neighbors = get_neighbors(current_node, grid)
        for neighbor_position in neighbors:
            if neighbor_position in closed_list:
                continue

            neighbor_node = Node(neighbor_position, current_node)
            neighbor_node.g = current_node.g + 1
            neighbor_node.h = heuristic(neighbor_node.postion, goal_node.postion)
            neighbor_node.f = neighbor_node.g + neighbor_node.h

            if add_to_open_list(open_list, neighbor_node):
                heapq.heappush(open_list, neighbor_node)
