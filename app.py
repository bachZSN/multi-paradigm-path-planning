import pygame
from environments.grid_world import Agent, create_default_world
from visualization.renderer import Renderer

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

        # Initialize the renderer
        renderer = Renderer(world)


        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            renderer.render_world(world, agents)
            self.clock.tick(self.FPS)

        pygame.quit()
