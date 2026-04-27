import pygame

class Renderer:
    def __init__(self, world, screen):
        self.world = world
        self.screen = screen
        self.grid_width = world.width
        self.grid_height = world.height
        self.screen_width = screen.get_width()
        self.screen_height = screen.get_height()
        self.margin = 50
        self.render_dimension = min(self.screen_width, self.screen_height) - 2 * self.margin
        self.cell_size = self.render_dimension // self.grid_width
        self.render_dimension = self.cell_size * self.grid_width  # Adjust render dimension to fit whole cells
        self.render_area = pygame.Rect(self.margin, self.margin, self.render_dimension, self.render_dimension)

    def render_world(self, grid_world, agents=None):
        # Fill the render area with a background color
        pygame.draw.rect(self.screen, (255, 255, 255), self.render_area)

        # Draw the grid, obstacles, and agents within the render area
        self.draw_grid(grid_world)
        self.draw_obstacles(grid_world)

        if agents:
            for agent in agents:
                self.draw_agent(agent)

    def draw_grid(self, grid_world):
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                rect = pygame.Rect(
                    self.render_area.x + x * self.cell_size,  # Offset by render_area.x
                    self.render_area.y + y * self.cell_size,  # Offset by render_area.y
                    self.cell_size,
                    self.cell_size
                )
                pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)

    def draw_obstacles(self, grid_world):
        for x, y in grid_world.obstacles:
            rect = pygame.Rect(
                self.render_area.x + x * self.cell_size,  # Offset by render_area.x
                self.render_area.y + y * self.cell_size,  # Offset by render_area.y
                self.cell_size,
                self.cell_size
            )
            pygame.draw.rect(self.screen, (0, 0, 0), rect)

    def draw_agent(self, agent):
        # Draw start position
        start_rect = pygame.Rect(
            self.render_area.x + agent.start[0] * self.cell_size,  # Offset by render_area.x
            self.render_area.y + agent.start[1] * self.cell_size,  # Offset by render_area.y
            self.cell_size,
            self.cell_size
        )
        pygame.draw.rect(self.screen, (0, 255, 0), start_rect)

        # Draw goal position
        goal_rect = pygame.Rect(
            self.render_area.x + agent.goal[0] * self.cell_size,  # Offset by render_area.x
            self.render_area.y + agent.goal[1] * self.cell_size,  # Offset by render_area.y
            self.cell_size,
            self.cell_size
        )
        pygame.draw.rect(self.screen, (255, 0, 0), goal_rect)

    def draw_path(self, path):
        for position in path:
            rect = pygame.Rect(
                self.render_area.x + position[0] * self.cell_size,  # Offset by render_area.x
                self.render_area.y + position[1] * self.cell_size,  # Offset by render_area.y
                self.cell_size,
                self.cell_size
            )
            pygame.draw.rect(self.screen, (0, 0, 255), rect)  # Blue for the path
