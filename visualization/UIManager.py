import pygame

class UIManager:
    def __init__(self, screen, reset_world):
        self.screen = screen
        self.reset_world = reset_world
        self.buttons = []
        self.margin = 50  # Preset margin for spacing
        self.create_layout()

    def create_layout(self):
        screen_width = self.screen.get_width()

        # Calculate dynamic button height and spacing
        button_width = 200
        button_height = 80
        spacing = self.margin

        # Calculate starting y-coordinate for the first button
        start_y = self.margin

        # Add algorithm buttons
        self.add_button(screen_width - button_width - self.margin, start_y, button_width, button_height, (0, 128, 0), "A*", self.astar_action)
        self.add_button(screen_width - button_width - self.margin, start_y + (button_height + spacing), button_width, button_height, (0, 128, 0), "Diffusion", self.diffusion_action)
        self.add_button(screen_width - button_width - self.margin, start_y + 2 * (button_height + spacing), button_width, button_height, (0, 128, 0), "Hill Climb", self.hill_climbing_action)

        # Add Reset and Quit buttons
        self.add_button(screen_width - button_width - self.margin, start_y + 3 * (button_height + spacing), button_width, button_height, (128, 0, 0), "Reset", self.reset_action, hover_color=(150, 0, 0))
        self.add_button(screen_width - button_width - self.margin, start_y + 4 * (button_height + spacing), button_width, button_height, (128, 0, 0), "Quit", self.quit_action, hover_color=(150, 0, 0))

        # Add control buttons with icons (repositioned to avoid overlap)
        icon_button_size = 50
        self.add_icon_button(screen_width - button_width - self.margin - 2 * icon_button_size - 3 * spacing, start_y + 4 * (button_height + spacing), icon_button_size, icon_button_size, (128, 128, 128), "<", self.backward_action)
        self.add_icon_button(screen_width - button_width - self.margin - icon_button_size - 2 * spacing, start_y + 4 * (button_height + spacing), icon_button_size, icon_button_size, (128, 128, 128), ">", self.forward_action)
        self.add_icon_button(screen_width - button_width - self.margin - icon_button_size - 2 * spacing, start_y + 3 * (button_height + spacing), icon_button_size, icon_button_size, (128, 128, 128), "▶", self.play_action)
        self.add_icon_button(screen_width - button_width - self.margin - 2 * icon_button_size - 3 * spacing, start_y + 3 * (button_height + spacing), icon_button_size, icon_button_size, (128, 128, 128), "||", self.pause_action)

    def add_button(self, x, y, width, height, color, text, callback, hover_color=None):
        """Add a text-based button to the UI."""
        button = {
            "rect": pygame.Rect(x, y, width, height),
            "color": color,
            "hover_color": hover_color if hover_color else (0, 150, 0),  # Default hover color is green
            "text": text,
            "callback": callback
        }
        self.buttons.append(button)

    def add_icon_button(self, x, y, width, height, color, icon, callback):
        """Add an icon-based button to the UI."""
        button = {
            "rect": pygame.Rect(x, y, width, height),
            "color": color,
            "hover_color": (150, 150, 150),  # Color when hovered
            "icon": icon,
            "callback": callback
        }
        self.buttons.append(button)

    def draw_buttons(self):
        """Draw all buttons on the screen with hover interaction."""
        mouse_pos = pygame.mouse.get_pos()  # Get the current mouse position
        for button in self.buttons:
            # Check if the mouse is hovering over the button
            if button["rect"].collidepoint(mouse_pos):
                color = button["hover_color"]  # Use hover color
            else:
                color = button["color"]  # Use default color

            # Draw the rounded rectangle for the button
            pygame.draw.rect(self.screen, color, button["rect"], border_radius=15)

            # Draw the button text or icon
            font = pygame.font.Font(None, 36)
            if "text" in button:
                text_surface = font.render(button["text"], True, (255, 255, 255))
                text_rect = text_surface.get_rect(center=button["rect"].center)
                self.screen.blit(text_surface, text_rect)
            elif "icon" in button:
                icon_surface = font.render(button["icon"], True, (255, 255, 255))
                icon_rect = icon_surface.get_rect(center=button["rect"].center)
                self.screen.blit(icon_surface, icon_rect)

    def handle_event(self, event):
        """Handle events for the UI components."""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # Left mouse button
            for button in self.buttons:
                if button["rect"].collidepoint(event.pos):
                    button["callback"]()

    # Define actions for the algorithm buttons
    def astar_action(self):
        print("A* Algorithm selected!")

    def diffusion_action(self):
        print("Diffusion Algorithm selected!")

    def hill_climbing_action(self):
        print("Hill Climbing Algorithm selected!")

    # Define actions for the Reset and Quit buttons
    def reset_action(self):
        self.reset_world()

    def quit_action(self):
        pygame.quit()
        exit()

    # Define actions for the control buttons
    def backward_action(self):
        print("Step Backward")

    def forward_action(self):
        print("Step Forward")

    def play_action(self):
        print("Play")

    def pause_action(self):
        print("Pause")
