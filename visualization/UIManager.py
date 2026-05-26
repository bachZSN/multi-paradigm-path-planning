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

        # Button geometry
        # Keep both columns to the RIGHT of the legend.
        button_width = 160
        button_height = 80
        row_spacing = 10
        col_spacing = 20

        right_x = screen_width - button_width - self.margin
        left_x = right_x - col_spacing - button_width

        # Right column: diffusion variants (top -> bottom), anchored to bottom-right
        right_items = [
            ("Diffusion", lambda: self.ui_action("Diffusion")),
            ("Diffusion+", lambda: self.ui_action("Diffusion+")),
            ("Diffusion~", lambda: self.ui_action("Diffusion~")),
            ("Coord-Diff", lambda: self.ui_action("Coord-Diff")),
            ("Coord-Diff+", lambda: self.ui_action("Coord-Diff+")),
            ("Coord-Diff~", lambda: self.ui_action("Coord-Diff~")),
        ]
        right_total_h = len(right_items) * button_height + (len(right_items) - 1) * row_spacing
        right_start_y = screen_height - self.margin - right_total_h

        for i, (label, cb) in enumerate(right_items):
            y = right_start_y + i * (button_height + row_spacing)
            self.add_button(right_x, y, button_width, button_height, (0, 128, 0), label, cb)

        # Left column: app controls (top -> bottom), also anchored to bottom
        left_items = [
            ("A*", lambda: self.ui_action("A*"), (0, 128, 0)),
            ("Toggle Path", self.toggle_show_path, (128, 0, 0)),
            ("Reset", lambda: self.ui_action("Reset"), (128, 0, 0)),
            ("Quit", lambda: self.ui_action("Quit"), (128, 0, 0)),
        ]
        left_total_h = len(left_items) * button_height + (len(left_items) - 1) * row_spacing
        left_start_y = screen_height - self.margin - left_total_h

        for i, (label, cb, color) in enumerate(left_items):
            y = left_start_y + i * (button_height + row_spacing)
            hover = (0, 150, 0) if color == (0, 128, 0) else (150, 0, 0)
            self.add_button(left_x, y, button_width, button_height, color, label, cb, hover_color=hover)

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
            font = pygame.font.Font(None, 32)
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
                    case pygame.K_1:
                        self.ui_action("A*")
                    case pygame.K_2:
                        self.ui_action("Diffusion")
                    case pygame.K_3:
                        self.ui_action("Coord-Diff")
                    case pygame.K_4:
                        self.ui_action("Diffusion+")
                    case pygame.K_5:
                        self.ui_action("Coord-Diff+")
                    case pygame.K_6:
                        self.ui_action("Diffusion~")
                    case pygame.K_7:
                        self.ui_action("Coord-Diff~")
                    case pygame.K_SPACE:
                        pass
                    case pygame.K_RIGHT:
                        pass
                    case pygame.K_LEFT:
                        pass
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
