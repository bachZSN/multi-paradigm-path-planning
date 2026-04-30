import pygame
from environments.grid_world import Agent, create_default_world
from visualization.UIManager import UIManager
from visualization.renderer import Renderer
from algorithms.astar import astar, dijsktra_search, bfs

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
        self.ui_manager = UIManager(self.screen, self.ui_action)
        self.explored_path = None
        self.shortest_path = None
        self.show_path = True
        self.agent = Agent(start=(5, 5), goal=(84, 69))
        self.agents = [self.agent]

    def reset_world(self):
        self.world = create_default_world()

    def ui_action(self, action_name):
        match action_name:
            case "A*":
                self.explored_path, self.shortest_path = astar(self.agent.start, self.agent.goal, self.world)
            case "Diffusion":
                print ("Diffusion button clicked")
                self.explored_path, self.shortest_path = dijsktra_search(self.agent.start, self.agent.goal, self.world)
            case "Hill Climb":
                self.explored_path, self.shortest_path = bfs(self.agent.start, self.agent.goal, self.world)
                print ("Hill Climb button clicked")
            case "Toggle Path":
                self.show_path = not self.show_path
            case "Reset":
                self.reset_world()
            case "Quit":
                pygame.quit()
                exit()
            case _:
                print ("Unknown action:", action_name)

    def run(self):

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                self.ui_manager.handle_event(event)

            self.renderer.render_world(self.world, self.agents)
            if (self.explored_path or self.shortest_path) and self.show_path:
                self.renderer.draw_path(self.explored_path, self.shortest_path)

            self.ui_manager.draw_buttons()

            pygame.display.flip()
            self.clock.tick(self.FPS)

        pygame.quit()
