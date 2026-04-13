from environments.grid_world import create_default_world, Agent
from visualization.renderer import Renderer
import pygame

def main():
    # Create the grid world
    world = create_default_world()

    # Create an agent
    agent = Agent(start=(0, 0), goal=(19, 19))

    # Initialize the renderer
    renderer = Renderer(grid_width=20, grid_height=20)

    # Game loop to render the grid world
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Render the world and agent
        renderer.render_world(world, agent)

    pygame.quit()

if __name__ == "__main__":
    main()
