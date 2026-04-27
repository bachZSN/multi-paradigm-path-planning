import heapq

class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0  # Cost from start to current node
        self.h = 0  # Heuristic cost to goal
        self.f = 0  # Total cost

    def __lt__(self, other):
        return self.f < other.f

def astar(start, goal, grid):
    open_list = []
    closed_list = set()
    start_node = Node(start)
    goal_node = Node(goal)
    heapq.heappush(open_list, start_node)

    while open_list:
        current_node = heapq.heappop(open_list)
        closed_list.add(current_node.position)

        if current_node.position == goal_node.position:
            return reconstruct_path(current_node)

        neighbors = get_neighbors(current_node, grid)
        for neighbor_position in neighbors:
            if neighbor_position in closed_list:
                continue

            neighbor_node = Node(neighbor_position, current_node)
            neighbor_node.g = current_node.g + 1
            neighbor_node.h = heuristic(neighbor_node.position, goal_node.position)
            neighbor_node.f = neighbor_node.g + neighbor_node.h

            if add_to_open_list(open_list, neighbor_node):
                heapq.heappush(open_list, neighbor_node)

    return None

def get_neighbors(node, grid):
    neighbors = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
    for direction in directions:
        new_position = (node.position[0] + direction[0], node.position[1] + direction[1])
        if 0 <= new_position[0] < len(grid) and 0 <= new_position[1] < len(grid[0]) and grid[new_position[0]][new_position[1]] == 0:
            neighbors.append(new_position)
    return neighbors

def heuristic(position, goal):
    # Manhattan distance
    return abs(position[0] - goal[0]) + abs(position[1] - goal[1])

def add_to_open_list(open_list, neighbor):
    for node in open_list:
        if neighbor.position == node.position and neighbor.g >= node.g:
            return False
    return True

def reconstruct_path(node):
    path = []
    while node:
        path.append(node.position)
        node = node.parent
    return path[::-1]
