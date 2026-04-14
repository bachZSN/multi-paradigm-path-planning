import pygame
from environments.grid_world import Agent, create_default_world
from visualization.renderer import Renderer
from algorithms.astar import astar, grid_world_to_grid

class App:
    def __init__(self):
        self.clock = pygame.time.Clock()
        self.FPS = 60

    def run(self):

        # Create the grid world
        world = create_default_world()

        # Create an agent
        agent = Agent(start=(0, 0), goal=(19, 19))
        agents = [agent]

        # Run the A* algorithm
        grid = grid_world_to_grid(world)
        path = astar(agent.start, agent.goal, grid)
        print("Path found:", path)

        # Initialize the renderer
        renderer = Renderer(world)


        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            renderer.render_world(world, agents)
            if path:
                renderer.draw_path(path)

            pygame.display.flip()

            self.clock.tick(self.FPS)

        pygame.quit()
