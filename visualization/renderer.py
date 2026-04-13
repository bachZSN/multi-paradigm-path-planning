import pygame

class Renderer:
    def __init__(self, grid_width, grid_height, screen_width=800, screen_height=800):
        pygame.init()
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Grid World")

    def render_world(self, grid_world, agent=None):
        """Render the grid world with obstacles and agent."""
        grid_data = [[0 for _ in range(grid_world.width)] for _ in range(grid_world.height)]

        # Add obstacles
        for x, y in grid_world.obstacles:
            grid_data[y][x] = 1

        # Add agent
        if agent:
            grid_data[agent.start[1]][agent.start[0]] = 2  # start
            grid_data[agent.goal[1]][agent.goal[0]] = 3    # goal

        self.render(grid_data)

    def render(self, grid_data):
        """Render grid data using pygame."""
        self.screen.fill((255, 255, 255))

        cell_width = self.screen_width // self.grid_width
        cell_height = self.screen_height // self.grid_height

        for y, row in enumerate(grid_data):
            for x, cell in enumerate(row):
                rect = pygame.Rect(x * cell_width, y * cell_height, cell_width, cell_height)

                if cell == 0:  # empty
                    color = (255, 255, 255)
                elif cell == 1:  # obstacle
                    color = (0, 0, 0)
                elif cell == 2:  # start
                    color = (0, 255, 0)
                elif cell == 3:  # goal
                    color = (255, 0, 0)
                else:
                    color = (255, 255, 255)

                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)

        pygame.display.flip()

# Example usage
if __name__ == "__main__":
    from environments.grid_world import create_default_world, Agent

    grid_world = create_default_world()
    agent = Agent(start=(0, 0), goal=(19, 19))

    renderer = Renderer(grid_width=20, grid_height=20)
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        renderer.render_world(grid_world, agent)

    pygame.quit()
