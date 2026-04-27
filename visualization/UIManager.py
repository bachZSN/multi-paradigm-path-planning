import pygame

class UIManager:
    def __init__(self, screen):
        self.screen = screen
        self.buttons = []
        self.create_layout()

    def create_layout(self):
        screen_width = self.screen.get_width()
        screen_height = self.screen.get_height()

        # Add buttons with consistent styling
        self.add_button(screen_width - 300, screen_height - 350, 200, 80, (0, 128, 0), "Start", self.start_action)
        self.add_button(screen_width - 300, screen_height - 250, 200, 80, (0, 128, 0), "Reset", self.reset_action)
        self.add_button(screen_width - 300, screen_height - 150, 200, 80, (0, 128, 0), "Quit", self.quit_action)

    def add_button(self, x, y, width, height, color, text, callback):
        """Add a button to the UI."""
        button = {
            "rect": pygame.Rect(x, y, width, height),
            "color": color,
            "hover_color": (0, 150, 0),  # Color when hovered
            "text": text,
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

            # Draw the button text
            font = pygame.font.Font(None, 36)
            text_surface = font.render(button["text"], True, (255, 255, 255))
            text_rect = text_surface.get_rect(center=button["rect"].center)
            self.screen.blit(text_surface, text_rect)

    def handle_event(self, event):
        """Handle events for the UI components."""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # Left mouse button
            for button in self.buttons:
                if button["rect"].collidepoint(event.pos):
                    button["callback"]()

    # Define actions for the buttons
    def start_action(self):
        print("Start button clicked!")

    def reset_action(self):
        print("Reset button clicked!")

    def quit_action(self):
        pygame.quit()
        exit()
