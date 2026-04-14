import pygame

class Renderer:
    def __init__(self, world, screen_width=800, screen_height=800):
        pygame.init()
        self.grid_width = world.width
        self.grid_height = world.height
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.cell_size = min(screen_width // world.width, screen_height // world.height)
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Grid World")

    def render_world(self, grid_world, agents=None):
        self.screen.fill((255, 255, 255))

        self.draw_grid(grid_world)
        self.draw_obstacles(grid_world)

        if agents:
            for agent in agents:
                self.draw_agent(agent)

    def draw_grid(self, grid_world):
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)

    def draw_obstacles(self, grid_world):
        for x, y in grid_world.obstacles:
            rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (0, 0, 0), rect)

    def draw_agent(self, agent):
        # Draw start position
        start_rect = pygame.Rect(agent.start[0] * self.cell_size, agent.start[1] * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, (0, 255, 0), start_rect)

        # Draw goal position
        goal_rect = pygame.Rect(agent.goal[0] * self.cell_size, agent.goal[1] * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, (255, 0, 0), goal_rect)

    def draw_path(self, path):
        for position in path:
            rect = pygame.Rect(position[0] * self.cell_size, position[1] * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (0, 0, 255), rect)
