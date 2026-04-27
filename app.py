import pygame
from environments.grid_world import Agent, create_default_world
from visualization.UIManager import UIManager
from visualization.renderer import Renderer
from algorithms.astar import astar, grid_world_to_grid

class App:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((1280, 720))
        pygame.display.set_caption("Planning Algorithms Visualization")
        self.screen.fill((255, 255, 255))
        self.clock = pygame.time.Clock()
        self.FPS = 60

    def run(self):

        # Create the grid world
        world = create_default_world()

        # Create an agent
        agent = Agent(start=(0, 0), goal=(79, 59))
        agents = [agent]

        # Run the A* algorithm
        grid = grid_world_to_grid(world)
        path = astar(agent.start, agent.goal, grid)
        print("Path found:", path)

        # Initialize the renderer
        renderer = Renderer(world, self.screen)
        ui_manager = UIManager(self.screen)


        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                ui_manager.handle_event(event)

            renderer.render_world(world, agents)
            if path:
                renderer.draw_path(path)

            ui_manager.draw_buttons()

            pygame.display.flip()

            self.clock.tick(self.FPS)

        pygame.quit()
