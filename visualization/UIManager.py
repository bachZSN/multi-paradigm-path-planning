import pygame

class UIManager:
    def __init__(self, screen, ui_action):
        self.screen = screen
        self.ui_action = ui_action
        self.buttons = []
        self.margin = 50  # Preset margin for spacing
        self.is_playing = False
        self.is_showing_path = True
        self.create_layout()

    def create_layout(self):
        screen_width = self.screen.get_width()
        screen_height = self.screen.get_height()

        # Calculate dynamic button height and spacing
        button_width = 200
        button_height = 80
        spacing = self.margin

        # Calculate starting y-coordinate for the first button
        start_y = self.margin

        # Add algorithm buttons
        self.add_button(screen_width - button_width - self.margin, start_y, button_width, button_height, (0, 128, 0), "A*", lambda: self.ui_action("A*"))
        self.add_button(screen_width - button_width - self.margin, start_y + (button_height + spacing), button_width, button_height, (0, 128, 0), "Diffusion", lambda: self.ui_action("Diffusion"))
        self.add_button(screen_width - button_width - self.margin, start_y + 2 * (button_height + spacing), button_width, button_height, (0, 128, 0), "Hill Climb", lambda: self.ui_action("Hill Climb"))

        # Add Reset and Quit buttons
        self.add_button(screen_width - 2* button_width - 2 * self.margin , start_y + 3 * (button_height + spacing), button_width, button_height, (128, 0, 0), "Toggle Path", self.toggle_show_path, hover_color=(150, 0, 0))
        self.add_button(screen_width - button_width - self.margin, start_y + 3 * (button_height + spacing), button_width, button_height, (128, 0, 0), "Reset", lambda: self.ui_action("Reset"), hover_color=(150, 0, 0))
        self.add_button(screen_width - button_width - self.margin, start_y + 4 * (button_height + spacing), button_width, button_height, (128, 0, 0), "Quit", lambda: self.ui_action("Quit"), hover_color=(150, 0, 0))

        # Add control buttons with icons (repositioned to avoid overlap)
        icon_button_size = 50
        self.add_icon_button(screen_width - button_width - self.margin - 3 * icon_button_size - 3 * spacing, start_y + 4 * (button_height + spacing), icon_button_size, icon_button_size, (128, 128, 128), "<", lambda: self.ui_action("Backward"))
        self.add_icon_button(screen_width - button_width - self.margin - 2 * icon_button_size - 2 * spacing, start_y + 4 * (button_height + spacing), icon_button_size, icon_button_size, (128, 128, 128), ">", lambda: self.ui_action("Forward"))
        self.add_icon_button(screen_width - button_width - self.margin - icon_button_size - 1 * spacing, start_y + 4 * (button_height + spacing), icon_button_size, icon_button_size, (128, 128, 128), "►", self.toggle_play_pause)

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
        match event.type:
            case pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    for button in self.buttons:
                        if button["rect"].collidepoint(event.pos):
                            button["callback"]()

            case pygame.KEYDOWN:
                match event.key:
                    case pygame.K_SPACE:
                        self.toggle_play_pause()
                    case pygame.K_RIGHT:
                        self.ui_action("Forward")
                    case pygame.K_LEFT:
                        self.ui_action("Backward")
                    case pygame.K_t:
                        self.toggle_show_path()
                    case pygame.K_r:
                        self.ui_action("Reset")
                    case pygame.K_q:
                        self.ui_action("Quit")
                    case _:
                        print(f"Unhandled key press: {pygame.key.name(event.key)}")

            case pygame.QUIT:
                pygame.quit()
                exit()

            case _:
                pass
                #print(f"Unhandled event: {event}")

    def toggle_play_pause(self):
        self.is_playing = not self.is_playing
        for button in self.buttons:
            if "icon" in button and button["icon"] in ["►", "||"]:
                button["icon"] = "||" if self.is_playing else "►"
                self.ui_action("Play" if self.is_playing else "Pause")
                break

    def toggle_show_path(self):
        self.is_showing_path = not self.is_showing_path
        self.ui_action("Toggle Path")
