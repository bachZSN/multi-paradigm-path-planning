import os
import torch
import pygame
from environments.grid_world import Agent, create_default_world
from visualization.UIManager import UIManager
from visualization.renderer import Renderer
from algorithms.astar import astar, dijkstra_search, bfs, calculate_total_cost
from algorithms.diffusion import PathUNet, infer_path
from algorithms.diffusion_coord import TrajectoryUNet1D, infer_path_coord

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
        self.current_cost = None
        self.total_cost = None
        self.explored_path = None
        self.shortest_path = None
        self.show_path = True
        self.agent = Agent(start=(92, 5), goal=(7, 75))
        self.agents = [self.agent]

    def reset_world(self):
        self.world = create_default_world()
        self.current_cost = None
        self.total_cost = 0
        self.explored_path = None
        self.shortest_path = None


    def ui_action(self, action_name):
        match action_name:
            case "A*":
                self.explored_path, self.shortest_path = astar(self.agent.start, self.agent.goal, self.world)
            case "Diffusion":
                print ("Running trained diffusion model ...")
                ckpt = "data/diffusion_model.pt"
                if not os.path.exists(ckpt):
                    print(f"  No checkpoint found at {ckpt}. Train one with: python -m experiments.train_diffusion")
                    return
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = PathUNet(in_channels=4, out_channels=1, time_dim=256)
                explored, shortest = infer_path(model, self.world, self.agent.start,
                                                self.agent.goal, checkpoint=ckpt, device=device)
                self.explored_path = explored if explored else None
                self.shortest_path = shortest if shortest else None
                print(f"  Diffusion path found: {len(self.shortest_path) if self.shortest_path else 0} waypoints")
            case "Coord-Diff":
                print ("Running coordinate trajectory diffusion model ...")
                ckpt = "data/diffusion_coord_model.pt"
                if not os.path.exists(ckpt):
                    print(f"  No checkpoint found at {ckpt}. Train the coordinate model first.")
                    return
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = TrajectoryUNet1D(time_dim=128, T=64)
                explored, shortest = infer_path_coord(model, self.world, self.agent.start,
                                                       self.agent.goal, checkpoint=ckpt, device=device)
                self.explored_path = explored if explored else None
                self.shortest_path = shortest if shortest else None
                print(f"  Coord-Diff path found: {len(self.shortest_path) if self.shortest_path else 0} waypoints")
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
                self.renderer.draw_cost(calculate_total_cost(self.shortest_path, self.world))

            self.ui_manager.draw_buttons()

            pygame.display.flip()
            self.clock.tick(self.FPS)

        pygame.quit()
