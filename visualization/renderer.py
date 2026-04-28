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

        # Draw the grid as a height map
        self.draw_height_map(grid_world)

        # Draw agents
        if agents:
            for agent in agents:
                self.draw_agent(agent)

        # Draw the height legend
        self.draw_legend(grid_world)

    def draw_height_map(self, grid_world):
        """Draw the grid as a height map using interval-based colors."""
        max_height = grid_world.max_height  # Get the maximum height in the grid
        min_height = grid_world.min_height  # Get the minimum height in the grid

        for x in range(self.grid_width):
            for y in range(self.grid_height):
                # Get the height value for the current cell
                height = grid_world.grid[y, x]

                # Determine the color based on the height
                if height == -1:  # Invalid height
                    color = (0, 0, 139)  # Dark blue
                else:
                    color = self.get_color_from_height(height, min_height, max_height)

                # Draw the cell with the corresponding color
                rect = pygame.Rect(
                    self.render_area.x + x * self.cell_size,
                    self.render_area.y + y * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                pygame.draw.rect(self.screen, color, rect)

    def get_color_from_height(self, height, min_height, max_height):
        """Map a height value to one of several interval-based colors using a single match block."""
        # Handle the case where all heights are the same
        if max_height == min_height:
            return (34, 139, 34)  # Default to dark green if all heights are the same

        # Calculate the height intervals
        interval = (max_height - min_height) / 5
        thresholds = [
            min_height + interval,  # 1/5
            min_height + 2 * interval,  # 2/5
            min_height + 3 * interval,  # 3/5
            min_height + 4 * interval,  # 4/5
            max_height  # Max height
        ]

        # Use a single match block to handle all cases
        match height:
            case -1:
                return (0, 0, 139)  # Dark Blue for obstacles
            case float('inf'):
                return (128, 128, 128)  # Gray for impassable cells
            case h if h <= thresholds[0]:
                return (0, 100, 0)  # Dark Green
            case h if h <= thresholds[1]:
                return (34, 139, 34)  # Green
            case h if h <= thresholds[2]:
                return (255, 255, 0)  # Yellow
            case h if h <= thresholds[3]:
                return (210, 180, 140)  # Light Brown
            case _:
                return (139, 69, 19)  # Brown

    def draw_legend(self, grid_world):
        """Draw a legend explaining the height colors with number intervals."""
        # Get the min and max heights from the grid
        max_height = grid_world.grid.max()
        min_height = grid_world.grid.min()

        # Define the height intervals
        interval = (max_height - min_height) / 5
        thresholds = [
            min_height,
            min_height + interval,
            min_height + 2 * interval,
            min_height + 3 * interval,
            min_height + 4 * interval,
            max_height
        ]

        # Define the legend colors
        legend_colors = [
            (0, 100, 0),    # Dark Green
            (34, 139, 34),  # Green
            (255, 255, 0),  # Yellow
            (210, 180, 140),  # Light Brown
            (139, 69, 19),  # Brown
            (0, 0, 139),    # Dark Blue for water
            (128, 128, 128) # Gray for impassable cells
        ]
        legend_labels = [
            f"{thresholds[0]:.1f} - {thresholds[1]:.1f}",
            f"{thresholds[1]:.1f} - {thresholds[2]:.1f}",
            f"{thresholds[2]:.1f} - {thresholds[3]:.1f}",
            f"{thresholds[3]:.1f} - {thresholds[4]:.1f}",
            f"{thresholds[4]:.1f} - {thresholds[5]:.1f}",
            "Water",
            "Impassable"
        ]

        # Calculate the size of the legend dynamically
        font = pygame.font.Font(None, 24)
        item_height = 30  # Height of each legend item
        num_items = len(legend_colors) - 1  # Exclude the invalid height color
        content_height = (num_items + 1) * item_height  # Total height of the legend content
        content_width = 150  # Fixed width for the legend content
        margin = 10  # Small margin around the content

        # Calculate the legend background dimensions
        legend_x = self.render_area.right + self.margin
        legend_y = self.margin
        legend_width = content_width + 2 * margin
        legend_height = content_height + 2 * margin

        # Draw the legend background
        pygame.draw.rect(self.screen, (240, 240, 240), (legend_x, legend_y, legend_width, legend_height))

        # Draw the legend items
        for i, (color, label) in enumerate(zip(legend_colors, legend_labels)):
            rect_y = legend_y + margin + i * item_height  # Position of the color box
            pygame.draw.rect(self.screen, color, (legend_x + margin, rect_y, 20, 20))  # Color box
            text_surface = font.render(label, True, (0, 0, 0))  # Black text
            self.screen.blit(text_surface, (legend_x + margin + 30, rect_y))  # Text next to the color box

    def draw_agent(self, agent):
        # Draw start position
        start_rect = pygame.Rect(
            self.render_area.x + agent.start[0] * self.cell_size,
            self.render_area.y + agent.start[1] * self.cell_size,
            self.cell_size,
            self.cell_size
        )
        pygame.draw.rect(self.screen, (0, 255, 0), start_rect)

        # Draw goal position
        goal_rect = pygame.Rect(
            self.render_area.x + agent.goal[0] * self.cell_size,
            self.render_area.y + agent.goal[1] * self.cell_size,
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
