import os
import pygame
import time
from environments.grid_world import Agent, create_default_world
from visualization.UIManager import UIManager
from visualization.renderer import Renderer
from algorithms.astar import astar, dijkstra_search, bfs, calculate_total_cost

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
        self.agent = Agent(start=(92, 5), goal=(2, 85))
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
            case "Diffusion" | "Diffusion+" | "Diffusion~":
                refine = {
                    "Diffusion": "dijkstra",
                    "Diffusion+": "astar_model",
                    "Diffusion~": "greedy_model",
                }[action_name]

                print(f"Running trained diffusion model ({refine}) ...")
                import torch
                from algorithms.diffusion import PathUNet, infer_path
                ckpt = "data/diffusion_model.pt"
                if not os.path.exists(ckpt):
                    print(f"  No checkpoint found at {ckpt}. Train one with: python -m experiments.train_diffusion")
                    return

                # Baseline A* for timing/cost comparison
                t0 = time.perf_counter()
                _explored_astar, path_astar = astar(self.agent.start, self.agent.goal, self.world)
                t1 = time.perf_counter()
                cost_astar = calculate_total_cost(path_astar, self.world) if path_astar else None

                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = PathUNet(in_channels=4, out_channels=1, time_dim=256)

                metrics = {}
                explored, shortest = infer_path(
                    model,
                    self.world,
                    self.agent.start,
                    self.agent.goal,
                    checkpoint=ckpt,
                    device=device,
                    metrics=metrics,
                    refine=refine,
                )
                self.explored_path = explored if explored else None
                self.shortest_path = shortest if shortest else None
                print(f"  Diffusion path found: {len(self.shortest_path) if self.shortest_path else 0} waypoints")

                cost_grid = calculate_total_cost(self.shortest_path, self.world) if self.shortest_path else None

                print("  Metrics:")
                print(f"    A* time:        {(t1 - t0) * 1000.0:.2f} ms")
                if _explored_astar is not None:
                    print(f"    A* discovered:  {len(_explored_astar)}")
                if cost_astar is not None:
                    print(f"    A* cost:        {cost_astar:.2f}")
                if "grid.sample_seconds" in metrics:
                    print(f"    Diffuse sample: {metrics['grid.sample_seconds'] * 1000.0:.2f} ms")
                if "grid.refine_seconds" in metrics:
                    print(f"    Refine search:  {metrics['grid.refine_seconds'] * 1000.0:.2f} ms")
                if "grid.total_seconds" in metrics:
                    print(f"    Total:          {metrics['grid.total_seconds'] * 1000.0:.2f} ms")
                if "grid.refine_expanded" in metrics:
                    print(f"    Refine expanded:{metrics['grid.refine_expanded']}")
                if "grid.refine_pushed" in metrics:
                    print(f"    Refine pushed:  {metrics['grid.refine_pushed']}")
                if "grid.refine_discovered" in metrics:
                    print(f"    Refine discover:{metrics['grid.refine_discovered']}")
                if cost_grid is not None:
                    print(f"    Diffusion cost: {cost_grid:.2f}")
                if (cost_astar is not None) and (cost_grid is not None):
                    print(f"    Cost delta:     {cost_grid - cost_astar:+.2f} (Diffusion - A*)")
            case "Coord-Diff" | "Coord-Diff+" | "Coord-Diff~":
                refine = {
                    "Coord-Diff": "dijkstra",
                    "Coord-Diff+": "astar_model",
                    "Coord-Diff~": "greedy_model",
                }[action_name]

                print(f"Running coordinate trajectory diffusion model ({refine}) ...")
                import torch
                from algorithms.diffusion_coord import TrajectoryUNet1D, infer_path_coord
                ckpt = "data/diffusion_coord_model.pt"
                if not os.path.exists(ckpt):
                    print(f"  No checkpoint found at {ckpt}. Train the coordinate model first.")
                    return

                # Baseline A* for timing/cost comparison
                t0 = time.perf_counter()
                _explored_astar, path_astar = astar(self.agent.start, self.agent.goal, self.world)
                t1 = time.perf_counter()
                cost_astar = calculate_total_cost(path_astar, self.world) if path_astar else None

                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = TrajectoryUNet1D(time_dim=128, T=64)

                metrics = {}
                explored, shortest = infer_path_coord(
                    model,
                    self.world,
                    self.agent.start,
                    self.agent.goal,
                    checkpoint=ckpt,
                    device=device,
                    metrics=metrics,
                    refine=refine,
                )
                self.explored_path = explored if explored else None
                self.shortest_path = shortest if shortest else None
                print(f"  Coord-Diff path found: {len(self.shortest_path) if self.shortest_path else 0} waypoints")

                cost_coord = calculate_total_cost(self.shortest_path, self.world) if self.shortest_path else None

                print("  Metrics:")
                print(f"    A* time:        {(t1 - t0) * 1000.0:.2f} ms")
                if _explored_astar is not None:
                    print(f"    A* discovered:  {len(_explored_astar)}")
                if cost_astar is not None:
                    print(f"    A* cost:        {cost_astar:.2f}")
                if "coord.sample_seconds" in metrics:
                    print(f"    Diffuse sample: {metrics['coord.sample_seconds'] * 1000.0:.2f} ms")
                if "coord.refine_seconds" in metrics:
                    print(f"    Refine search:  {metrics['coord.refine_seconds'] * 1000.0:.2f} ms")
                if "coord.total_seconds" in metrics:
                    print(f"    Total:          {metrics['coord.total_seconds'] * 1000.0:.2f} ms")
                if "coord.refine_expanded" in metrics:
                    print(f"    Refine expanded:{metrics['coord.refine_expanded']}")
                if "coord.refine_pushed" in metrics:
                    print(f"    Refine pushed:  {metrics['coord.refine_pushed']}")
                if "coord.refine_discovered" in metrics:
                    print(f"    Refine discover:{metrics['coord.refine_discovered']}")
                if cost_coord is not None:
                    print(f"    Coord cost:     {cost_coord:.2f}")
                if (cost_astar is not None) and (cost_coord is not None):
                    print(f"    Cost delta:     {cost_coord - cost_astar:+.2f} (Coord - A*)")
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
