import pygame
from environments.grid_world import Agent, create_default_world
from visualization.UIManager import UIManager
from visualization.renderer import Renderer
from algorithms.astar import astar

class App:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((1280, 720))
        pygame.display.set_caption("Planning Algorithms Visualization")
        self.screen.fill((255, 255, 255))
        self.clock = pygame.time.Clock()
        self.FPS = 60
        self.world = create_default_world()
        self.renderer = Renderer(self.world, self.screen)
        self.ui_manager = UIManager(self.screen, self.reset_world)

    def reset_world(self):
        self.world = create_default_world()

    def run(self):

        # Create an agent
        agent = Agent(start=(5, 5), goal=(84, 69))
        agents = [agent]

        # Run the A* algorithm
        path = None
        #path = astar(agent.start, agent.goal, grid)
        #print("Path found:", path)

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                self.ui_manager.handle_event(event)

            self.renderer.render_world(self.world, agents)
            if path:
                self.renderer.draw_path(path)

            self.ui_manager.draw_buttons()

            pygame.display.flip()
            self.clock.tick(self.FPS)

        pygame.quit()
