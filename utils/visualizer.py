"""
SPACE VISUALIZER

This module makes the robot's learning journey VISIBLE and FUN!
Using pygame, we draw a colorful space station where the robot moves around.

Kids can watch the robot learn in real-time and see:
- The robot moving (smooth animations!)
- Energy stars being collected
- Asteroids to avoid
- The robot's "brain" (Q-values) in action
- Learning progress over time
"""

import pygame
import numpy as np
from typing import Optional, Dict, Tuple, List
import sys

# Initialize pygame
pygame.init()

# SPACE COLOR PALETTE
colors = {
    'space_dark': (10, 14, 39),      # Deep space blue
    'space_medium': (26, 31, 58),    # Medium space blue
    'space_light': (45, 55, 90),     # Light space blue
    'robot': (0, 212, 255),          # Bright cyan (robot glow)
    'robot_dark': (0, 150, 200),     # Darker cyan
    'asteroid': (255, 71, 87),       # Red-orange
    'asteroid_dark': (200, 50, 60),  # Darker red
    'energy': (255, 215, 0),         # Gold
    'energy_glow': (255, 255, 150),  # Light gold
    'goal': (46, 204, 113),          # Green
    'goal_glow': (100, 255, 150),    # Light green
    'grid': (70, 80, 120),           # Grid lines
    'white': (255, 255, 255),
    'yellow': (255, 255, 0),
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (100, 150, 255),
    'text': (200, 210, 230),         # Light gray-blue text
}


class SpaceVisualizer:
    """
    Visualizes the space station and robot learning!
    
    Creates a pygame window showing:
    - The grid world with robot, asteroids, energy, and goal
    - Real-time stats and information
    - Q-value visualizations (the robot's brain)
    - Learning progress
    
    Perfect for kids to SEE reinforcement learning in action!
    """
    
    def __init__(self, grid_size: int = 5, cell_size: int = 100, fps: int = 30, pause_between_steps: bool = False):
        """
        Set up the visual display.
        
        Args:
            grid_size: Size of the space station grid (5x5, 8x8, etc.)
            cell_size: How big is each grid square in pixels? (default: 100 for larger display)
            fps: How fast does the animation run? (frames per second)
            pause_between_steps: If True, wait for user input (SPACE or click) between each action
        """
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.fps = fps
        self.pause_between_steps = pause_between_steps
        
        # Calculate window size
        # Main grid + side panel for information + space for Q-values
        self.grid_pixel_size = grid_size * cell_size
        self.panel_width = 500  # Side panel width (increased from 400)
        self.window_width = self.grid_pixel_size + self.panel_width + 400  # Increased spacing to prevent overlap
        self.window_height = max(self.grid_pixel_size + 350, 950)  # Even more space for Q-values (was 600)
        
        # Create the window
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Space Explorer - Robot Learning!")
        
        # Create a clock to control animation speed
        self.clock = pygame.time.Clock()
        
        # Load fonts
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 24)
        
        # Animation state
        self.robot_anim_pos = None  # For smooth robot movement
        self.particles = []  # For sparkle effects
        
        # Stats to display
        self.stats = {
            'episode': 0,
            'step': 0,
            'reward': 0,
            'total_reward': 0,
            'epsilon': 1.0,
            'energy_collected': 0,
        }
        
        # Q-table for visualization - stores full table for heatmap
        # Initialize with zeros, will be updated as robot visits states
        n_states = grid_size * grid_size
        n_actions = 4
        self.q_table = np.zeros((n_states, n_actions))  # Full Q-table for heatmap
        self.current_state = 0
        self.q_table_initialized = False
        
        # Track Q-values before and after learning for current state
        self.old_q_values = np.zeros(n_actions)  # Q-values before learning
        self.new_q_values = np.zeros(n_actions)  # Q-values after learning
        self.current_action = 0  # Action that was just taken
        self.is_exploration = False  # Whether current action was exploration or exploitation
        
    def _draw_grid(self, env_info: Dict):
        """
        Draw the grid background and cells.
        """
        # Fill background with space color
        self.screen.fill(colors['space_dark'])
        
        # Calculate grid offset to center it
        grid_x = 20
        grid_y = 80
        
        # Draw grid cells
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                x = grid_x + col * self.cell_size
                y = grid_y + row * self.cell_size
                
                # Draw cell background
                rect = pygame.Rect(x, y, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, colors['space_medium'], rect)
                
                # Draw cell border
                pygame.draw.rect(self.screen, colors['grid'], rect, 2)
        
        return grid_x, grid_y
    
    def _draw_robot(self, grid_x: int, grid_y: int, robot_pos: Tuple[int, int]):
        """
        Draw the robot at its current position.
        
        The robot is drawn as a cyan circle with a glow effect.
        """
        row, col = robot_pos
        
        # Calculate pixel position
        center_x = grid_x + col * self.cell_size + self.cell_size // 2
        center_y = grid_y + row * self.cell_size + self.cell_size // 2
        radius = self.cell_size // 3
        
        # Draw glow effect (outer circle)
        glow_radius = radius + 8
        glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        for i in range(5):
            alpha = 50 - i * 10
            pygame.draw.circle(
                glow_surface, 
                (*colors['robot'][:3], alpha),
                (glow_radius, glow_radius),
                glow_radius - i * 2
            )
        self.screen.blit(glow_surface, (center_x - glow_radius, center_y - glow_radius))
        
        # Draw robot body
        pygame.draw.circle(self.screen, colors['robot'], (center_x, center_y), radius)
        pygame.draw.circle(self.screen, colors['robot_dark'], (center_x, center_y), radius, 3)
        
        # Draw robot face (two eyes)
        eye_radius = 4
        left_eye = (center_x - 8, center_y - 2)
        right_eye = (center_x + 8, center_y - 2)
        pygame.draw.circle(self.screen, colors['white'], left_eye, eye_radius)
        pygame.draw.circle(self.screen, colors['white'], right_eye, eye_radius)
        pygame.draw.circle(self.screen, (0, 0, 0), left_eye, 2)
        pygame.draw.circle(self.screen, (0, 0, 0), right_eye, 2)
        
        # Draw smile
        smile_rect = pygame.Rect(center_x - 10, center_y + 2, 20, 10)
        pygame.draw.arc(self.screen, colors['white'], smile_rect, 3.14, 0, 2)
    
    def _draw_asteroids(self, grid_x: int, grid_y: int, asteroids: List[Tuple[int, int]]):
        """
        Draw asteroids (dangers to avoid).
        
        Asteroids are drawn as red-orange spiky circles.
        """
        for row, col in asteroids:
            x = grid_x + col * self.cell_size + self.cell_size // 2
            y = grid_y + row * self.cell_size + self.cell_size // 2
            radius = self.cell_size // 3
            
            # Draw spiky asteroid shape
            import math
            points = []
            num_spikes = 8
            for i in range(num_spikes * 2):
                angle = (i * math.pi) / num_spikes
                if i % 2 == 0:
                    r = radius + 5
                else:
                    r = radius - 3
                px = x + r * math.cos(angle)
                py = y + r * math.sin(angle)
                points.append((px, py))
            
            pygame.draw.polygon(self.screen, colors['asteroid'], points)
            pygame.draw.polygon(self.screen, colors['asteroid_dark'], points, 2)
    
    def _draw_energy_stars(self, grid_x: int, grid_y: int, energy_stars: List[Tuple[int, int]]):
        """
        Draw energy stars (collectibles).
        
        Energy stars are drawn as golden stars with a glow.
        """
        import math
        
        for row, col in energy_stars:
            x = grid_x + col * self.cell_size + self.cell_size // 2
            y = grid_y + row * self.cell_size + self.cell_size // 2
            
            # Draw glow
            glow_radius = 20
            glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(
                glow_surface,
                (*colors['energy_glow'][:3], 100),
                (glow_radius, glow_radius),
                glow_radius
            )
            self.screen.blit(glow_surface, (x - glow_radius, y - glow_radius))
            
            # Draw star shape
            points = []
            num_points = 5
            outer_radius = 15
            inner_radius = 7
            
            for i in range(num_points * 2):
                angle = (i * math.pi) / num_points - math.pi / 2
                if i % 2 == 0:
                    r = outer_radius
                else:
                    r = inner_radius
                px = x + r * math.cos(angle)
                py = y + r * math.sin(angle)
                points.append((px, py))
            
            pygame.draw.polygon(self.screen, colors['energy'], points)
            pygame.draw.polygon(self.screen, (200, 170, 0), points, 2)
    
    def _draw_goal(self, grid_x: int, grid_y: int, goal_pos: Tuple[int, int]):
        """
        Draw the goal (green portal).
        
        The goal is drawn as a glowing green circle.
        """
        row, col = goal_pos
        x = grid_x + col * self.cell_size + self.cell_size // 2
        y = grid_y + row * self.cell_size + self.cell_size // 2
        radius = self.cell_size // 3
        
        # Draw glow
        glow_radius = radius + 15
        glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        for i in range(4):
            alpha = 80 - i * 15
            pygame.draw.circle(
                glow_surface,
                (*colors['goal_glow'][:3], alpha),
                (glow_radius, glow_radius),
                glow_radius - i * 4
            )
        self.screen.blit(glow_surface, (x - glow_radius, y - glow_radius))
        
        # Draw goal portal
        pygame.draw.circle(self.screen, colors['goal'], (x, y), radius)
        pygame.draw.circle(self.screen, colors['goal_glow'], (x, y), radius - 5)
        pygame.draw.circle(self.screen, colors['goal'], (x, y), radius - 10)
        
        # Draw "G" for goal
        font = pygame.font.Font(None, 30)
        text = font.render("G", True, colors['white'])
        text_rect = text.get_rect(center=(x, y))
        self.screen.blit(text, text_rect)
    
    def _draw_stats_panel(self, grid_x: int, grid_y: int):
        """
        Draw the side panel with statistics.
        """
        panel_x = grid_x + self.grid_pixel_size + 30
        panel_y = grid_y
        
        # Draw panel background
        panel_rect = pygame.Rect(
            panel_x - 10, 
            panel_y - 10, 
            self.panel_width, 
            self.grid_pixel_size
        )
        pygame.draw.rect(self.screen, colors['space_medium'], panel_rect)
        pygame.draw.rect(self.screen, colors['grid'], panel_rect, 2)
        
        # Title
        title = self.font_large.render("Robot Stats", True, colors['yellow'])
        self.screen.blit(title, (panel_x, panel_y))
        
        # Stats
        y_offset = panel_y + 50
        stats_texts = [
            (f"Episode: {self.stats['episode']}", colors['white']),
            (f"Step: {self.stats['step']}", colors['white']),
            (f"Energy Collected: {self.stats['energy_collected']}", colors['energy']),
            (f"Reward: {self.stats['reward']:+.1f}", colors['green'] if self.stats['reward'] > 0 else colors['red']),
            (f"Total Score: {self.stats['total_reward']:.1f}", colors['white']),
            ("", colors['white']),
            (f"Exploration: {self.stats['epsilon']:.2f}", colors['blue']),
        ]
        
        for text, color in stats_texts:
            rendered = self.font_medium.render(text, True, color)
            self.screen.blit(rendered, (panel_x, y_offset))
            y_offset += 35
        
        # Legend
        y_offset += 30
        legend_title = self.font_medium.render("Legend:", True, colors['yellow'])
        self.screen.blit(legend_title, (panel_x, y_offset))
        y_offset += 35
        
        legend_items = [
            ("Robot", colors['robot']),
            ("Asteroid (DANGER!)", colors['asteroid']),
            ("Energy (+10 points)", colors['energy']),
            ("Goal (+100 points!)", colors['goal']),
        ]
        
        for text, color in legend_items:
            rendered = self.font_small.render(text, True, color)
            self.screen.blit(rendered, (panel_x, y_offset))
            y_offset += 30
    
    def _draw_arrow_shape(self, x: int, y: int, direction: int, size: int, color: tuple):
        """
        Draw an arrow as a shape (line with arrowhead) instead of Unicode character.
        
        Args:
            x, y: Center position of the arrow
            direction: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
            size: Size of the arrow
            color: RGB color tuple
        """
        half_size = size // 2
        quarter_size = size // 4
        
        if direction == 0:  # UP
            # Main line
            pygame.draw.line(self.screen, color, (x, y + half_size), (x, y - quarter_size), 3)
            # Arrowhead
            pygame.draw.polygon(self.screen, color, [
                (x, y - half_size),
                (x - quarter_size, y - quarter_size),
                (x + quarter_size, y - quarter_size)
            ])
        elif direction == 1:  # DOWN
            # Main line
            pygame.draw.line(self.screen, color, (x, y - half_size), (x, y + quarter_size), 3)
            # Arrowhead
            pygame.draw.polygon(self.screen, color, [
                (x, y + half_size),
                (x - quarter_size, y + quarter_size),
                (x + quarter_size, y + quarter_size)
            ])
        elif direction == 2:  # LEFT
            # Main line
            pygame.draw.line(self.screen, color, (x + half_size, y), (x - quarter_size, y), 3)
            # Arrowhead
            pygame.draw.polygon(self.screen, color, [
                (x - half_size, y),
                (x - quarter_size, y - quarter_size),
                (x - quarter_size, y + quarter_size)
            ])
        elif direction == 3:  # RIGHT
            # Main line
            pygame.draw.line(self.screen, color, (x - half_size, y), (x + quarter_size, y), 3)
            # Arrowhead
            pygame.draw.polygon(self.screen, color, [
                (x + half_size, y),
                (x + quarter_size, y - quarter_size),
                (x + quarter_size, y + quarter_size)
            ])
    
    def _draw_q_table_heatmap(self, grid_x: int, grid_y: int):
        """
        Q-TABLE HEATMAP - FULL visualization of robot's brain!
        
        Shows a color-coded grid where:
        - Color intensity = Best Q-value at that position (green = good, red = bad)
        - Arrow = Best action from that position (drawn as shape, not Unicode)
        - Numeric value = Best Q-value (shown for clarity)
        
        This helps kids understand where the robot thinks it's safe vs dangerous!
        """
        if self.q_table is None or len(self.q_table) == 0:
            return
        
        # Heatmap settings
        heatmap_title_y = grid_y + self.grid_pixel_size + 20
        heatmap_title_x = grid_x + self.grid_pixel_size + 180
        
        # Title (using ASCII only - no emojis)
        title = self.font_medium.render("Q-TABLE HEATMAP (Robot's Complete Brain):", True, colors['yellow'])
        self.screen.blit(title, (heatmap_title_x, heatmap_title_y))
        
        # Draw subtitle explaining the heatmap
        subtitle = self.font_small.render("Color = Best Q-value | Arrow = Best Action | Green=Good Red=Bad", True, colors['text'])
        self.screen.blit(subtitle, (heatmap_title_x, heatmap_title_y + 25))
        
        # Calculate heatmap cell size (smaller than main grid)
        heatmap_cell_size = min(45, (self.window_width - 100) // self.grid_size)
        heatmap_width = self.grid_size * heatmap_cell_size
        
        # Position heatmap below the subtitle
        heatmap_start_y = heatmap_title_y + 50
        heatmap_start_x = heatmap_title_x
        
        # Draw heatmap grid
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                # Calculate state number from position
                state = row * self.grid_size + col
                
                # Get Q-values for this state
                if state < len(self.q_table):
                    q_values = self.q_table[state]
                    best_q = np.max(q_values)
                    best_action = np.argmax(q_values)
                else:
                    best_q = 0
                    best_action = 0
                
                # Calculate heatmap cell position
                cell_x = heatmap_start_x + col * heatmap_cell_size
                cell_y = heatmap_start_y + row * heatmap_cell_size
                
                # Determine color based on best Q-value
                # Normalize: -50 to 100 range
                normalized_q = (best_q + 50) / 150
                normalized_q = max(0, min(1, normalized_q))
                
                # Calculate color
                if best_q > 50:
                    # Green (very good) - blend with intensity
                    intensity = int(100 + normalized_q * 155)
                    cell_color = (0, intensity, 100)  # Cyan-green
                elif best_q > 0:
                    # Yellow-green (good)
                    intensity = int(150 + normalized_q * 105)
                    cell_color = (intensity, intensity, 0)
                elif best_q > -20:
                    # White/gray (okay)
                    intensity = int(100 + normalized_q * 100)
                    cell_color = (intensity, intensity, intensity)
                else:
                    # Red (bad)
                    intensity = int(100 + (1 - normalized_q) * 155)
                    cell_color = (intensity, 50, 50)
                
                # Draw cell background
                cell_rect = pygame.Rect(cell_x, cell_y, heatmap_cell_size - 2, heatmap_cell_size - 2)
                pygame.draw.rect(self.screen, cell_color, cell_rect)
                pygame.draw.rect(self.screen, colors['grid'], cell_rect, 1)
                
                # Draw arrow for best action (using shape, not Unicode)
                arrow_x = cell_x + heatmap_cell_size // 2
                arrow_y = cell_y + heatmap_cell_size // 2
                arrow_size = heatmap_cell_size - 15  # Slightly smaller than cell
                self._draw_arrow_shape(arrow_x, arrow_y, int(best_action), arrow_size, colors['white'])
                
                # Draw numeric Q-value (small, for reference)
                value_font = pygame.font.Font(None, 14)
                value_text = value_font.render(f"{int(best_q)}", True, colors['white'])
                value_rect = value_text.get_rect(bottomright=(
                    cell_x + heatmap_cell_size - 4,
                    cell_y + heatmap_cell_size - 2
                ))
                self.screen.blit(value_text, value_rect)
        
        # Draw heatmap legend
        legend_y = heatmap_start_y + heatmap_width + 15
        legend_x = heatmap_title_x
        
        # Legend title
        legend_title = self.font_small.render("Quality Legend:", True, colors['white'])
        self.screen.blit(legend_title, (legend_x, legend_y))
        
        # Legend items - all with shape-based arrows (no Unicode)
        legend_items = [
            (0, colors['goal'], ">50 Excellent"),  # 0 = UP arrow
            (0, colors['energy'], ">0 Good"),
            (0, colors['white'], "-20 to 0 OK"),
            (0, colors['asteroid'], "<-20 Bad")
        ]
        
        legend_item_x = legend_x + 100
        for arrow_dir, color, label in legend_items:
            # Color box
            box_rect = pygame.Rect(legend_item_x, legend_y, 20, 20)
            pygame.draw.rect(self.screen, color, box_rect)
            pygame.draw.rect(self.screen, colors['white'], box_rect, 1)
            
            # Draw arrow shape (up arrow for all legend items)
            arrow_center_x = legend_item_x + 10
            arrow_center_y = legend_y + 10
            self._draw_arrow_shape(arrow_center_x, arrow_center_y, arrow_dir, 12, colors['white'])
            
            # Label
            label_surface = self.font_small.render(label, True, colors['text'])
            self.screen.blit(label_surface, (legend_item_x + 25, legend_y + 2))
            
            legend_item_x += 120
    
    def _draw_current_state_q_values(self, grid_x: int, grid_y: int):
        """
        Display Q-values for the CURRENT STATE only, showing before and after learning.
        
        Shows three columns:
        1. BEFORE LEARNING: Q-values before the update
        2. AFTER LEARNING: Q-values after the update
        3. CHANGE (Δ): Difference showing how much each Q-value changed
        
        This helps kids see exactly how Q-learning updates work step by step!
        """
        # Position below the main grid - use left side (share space with heatmap)
        display_y = grid_y + self.grid_pixel_size + 20
        display_x = grid_x
        
        # Title showing current state
        state_row = self.current_state // self.grid_size
        state_col = self.current_state % self.grid_size
        title_text = f"Q-VALUES FOR CURRENT STATE (State {self.current_state} - Position {state_row},{state_col}):"
        title = self.font_medium.render(title_text, True, colors['yellow'])
        self.screen.blit(title, (display_x, display_y))
        
        # Column headers
        col_y = display_y + 30
        headers = ["BEFORE LEARNING", "AFTER LEARNING", "CHANGE (Δ)"]
        col_width = 200
        
        for i, header in enumerate(headers):
            header_x = display_x + i * (col_width + 20)
            header_surface = self.font_small.render(header, True, colors['yellow'])
            self.screen.blit(header_surface, (header_x, col_y))
        
        # Draw action rows
        action_names = ["UP", "DOWN", "LEFT", "RIGHT"]
        row_y = col_y + 25
        row_height = 35
        
        for action_idx, action_name in enumerate(action_names):
            # Get Q-values
            old_val = self.old_q_values[action_idx] if action_idx < len(self.old_q_values) else 0
            new_val = self.new_q_values[action_idx] if action_idx < len(self.new_q_values) else 0
            delta = new_val - old_val
            
            # Determine if this is the action that was taken
            is_current_action = (action_idx == self.current_action)
            
            # Determine highlight color
            if is_current_action:
                bg_color = (255, 215, 0)  # Gold - current action
                text_color = colors['space_dark']
            else:
                bg_color = colors['space_medium']
                text_color = colors['white']
            
            # Draw row background
            row_rect = pygame.Rect(display_x, row_y, col_width * 3 + 40, row_height - 5)
            pygame.draw.rect(self.screen, bg_color, row_rect)
            
            # Action name (shared across all columns)
            action_surface = self.font_small.render(action_name, True, text_color if not is_current_action else colors['space_dark'])
            self.screen.blit(action_surface, (display_x + 5, row_y + 5))
            
            # Before value
            before_x = display_x + 60
            before_text = f"{old_val:6.1f}"
            before_surface = self.font_small.render(before_text, True, colors['white'])
            self.screen.blit(before_surface, (before_x, row_y + 5))
            
            # After value (highlight if it's the best)
            after_x = display_x + col_width + 20 + 60
            after_text = f"{new_val:6.1f}"
            # Check if this is the best action after learning
            if new_val == max(self.new_q_values):
                after_color = colors['goal']  # Green for best
            else:
                after_color = colors['white']
            after_surface = self.font_small.render(after_text, True, after_color)
            self.screen.blit(after_surface, (after_x, row_y + 5))
            
            # Change/Delta value
            delta_x = display_x + 2 * (col_width + 20) + 60
            if delta > 0:
                delta_color = colors['goal']  # Green for positive
                delta_text = f"+{delta:5.1f}"
            elif delta < 0:
                delta_color = colors['asteroid']  # Red for negative
                delta_text = f"{delta:6.1f}"
            else:
                delta_color = colors['text']  # Gray for zero
                delta_text = "  0.0"
            delta_surface = self.font_small.render(delta_text, True, delta_color)
            self.screen.blit(delta_surface, (delta_x, row_y + 5))
            
            # Add arrow indicator for current action
            if is_current_action:
                arrow_x = display_x - 10
                self._draw_arrow_shape(arrow_x, row_y + row_height//2, action_idx, 15, colors['yellow'])
            
            row_y += row_height
        
        # Draw explanation text
        explain_y = row_y + 10
        # Determine exploration/exploitation status
        if self.is_exploration:
            mode_text = "EXPLORATION (Random)"
            mode_color = colors['blue']
        else:
            mode_text = "EXPLOITATION (Best)"
            mode_color = colors['goal']
        
        explain_text = f"Action Taken: {action_names[self.current_action]} | "
        explain_text += f"Reward: {self.stats['reward']:.1f} | "
        explain_text += f"NewQ = OldQ + α(Reward + γ·Future - OldQ)"
        explain_surface = self.font_small.render(explain_text, True, colors['text'])
        self.screen.blit(explain_surface, (display_x, explain_y))
        
        # Render mode indicator separately with its color
        mode_indicator = f" [{mode_text}]"
        mode_surface = self.font_small.render(mode_indicator, True, mode_color)
        self.screen.blit(mode_surface, (display_x + 220, explain_y + 20))
    
    def _draw_q_values(self, grid_x: int, grid_y: int, q_values: Optional[np.ndarray] = None):
        """
        Draw the Q-values for the current state.
        
        Shows which moves the robot thinks are best from current position.
        (Kept for backwards compatibility, but heatmap is now preferred)
        """
        # Skip drawing individual Q-values if we have the heatmap
        # The heatmap provides more comprehensive information
        pass
    
    def update(
        self, 
        env_info: Dict,
        reward: float = 0,
        old_q_values: Optional[np.ndarray] = None,
        new_q_values: Optional[np.ndarray] = None,
        current_state: int = 0,
        current_action: int = 0,
        is_exploration: Optional[bool] = None,
    ):
        """
        Update the visual display with current information.
        
        This is the main function that draws everything.
        
        Args:
            env_info: Dictionary from the environment with positions
            reward: Current step's reward
            old_q_values: Q-values BEFORE learning (before update)
            new_q_values: Q-values AFTER learning (after update)
            current_state: Current state number
            current_action: Action that was just taken (0-3)
            is_exploration: True if action was exploration (random), False if exploitation (best action)
        """
        # Handle events (like closing the window)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                if event.key == pygame.K_SPACE:
                    # Pause - can be used for teaching mode
                    pass
        
        # Update stats
        self.stats['step'] = env_info.get('steps', 0)
        self.stats['reward'] = reward
        self.stats['total_reward'] += reward
        self.stats['energy_collected'] = env_info.get('collected_energy', 0)
        
        # Store current state and action
        self.current_state = current_state
        self.current_action = current_action
        
        # Store exploration/exploitation flag for display
        if is_exploration is not None:
            self.is_exploration = is_exploration
        
        # Store old and new Q-values for display
        if old_q_values is not None:
            self.old_q_values = old_q_values.copy()
        if new_q_values is not None:
            self.new_q_values = new_q_values.copy()
            # Also update the full Q-table for reference
            if current_state < len(self.q_table):
                self.q_table[current_state] = new_q_values.copy()
        
        # Get positions
        robot_pos = env_info.get('robot_pos', (0, 0))
        asteroids = env_info.get('asteroids', [])
        energy_stars = env_info.get('energy_stars', [])
        goal_pos = env_info.get('goal_pos', (4, 4))
        
        # Draw everything
        grid_x, grid_y = self._draw_grid(env_info)
        self._draw_asteroids(grid_x, grid_y, asteroids)
        self._draw_energy_stars(grid_x, grid_y, energy_stars)
        self._draw_goal(grid_x, grid_y, goal_pos)
        self._draw_robot(grid_x, grid_y, robot_pos)
        self._draw_stats_panel(grid_x, grid_y)
        # Draw current state Q-values (before/after learning) on the LEFT
        self._draw_current_state_q_values(grid_x, grid_y)
        # Draw full Q-table heatmap on the RIGHT (side by side)
        self._draw_q_table_heatmap(grid_x, grid_y)
        
        # Add title
        title = self.font_large.render("SPACE EXPLORER", True, colors['yellow'])
        self.screen.blit(title, (grid_x, 20))
        
        subtitle = self.font_small.render("Watch the robot learn!", True, colors['text'])
        self.screen.blit(subtitle, (grid_x, 55))
        
        # Update display
        pygame.display.flip()
        
        # Pause and wait for user input if enabled
        if self.pause_between_steps:
            self._wait_for_input()
        
        # Control speed
        self.clock.tick(self.fps)
    
    def _wait_for_input(self):
        """
        Pause and wait for user to press SPACE or click to continue.
        This is useful for teaching mode to explain each step.
        """
        waiting = True
        
        # Draw "waiting" message on screen
        wait_font = pygame.font.Font(None, 16)
        wait_text = wait_font.render("Press SPACE or Click to continue...", True, colors['yellow'])
        wait_rect = wait_text.get_rect(center=(self.window_width // 5, self.window_height - 50))
        self.screen.blit(wait_text, wait_rect)
        pygame.display.flip()
        
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()
                    if event.key == pygame.K_SPACE:
                        waiting = False  # Continue to next step
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    waiting = False  # Continue on mouse click
            
            # Small delay to prevent CPU hogging
            pygame.time.wait(10)
    
    def set_episode(self, episode: int):
        """
        Set the current episode number for display.
        """
        self.stats['episode'] = episode
        self.stats['total_reward'] = 0  # Reset for new episode
        # Only reset Q-table at episode 1 (fresh start)
        # For subsequent episodes, keep accumulating to show full learning progress
        if episode == 1:
            self.q_table = np.zeros((self.grid_size * self.grid_size, 4))
    
    def set_epsilon(self, epsilon: float):
        """
        Set the current exploration rate for display.
        """
        self.stats['epsilon'] = epsilon
    
    def close(self):
        """
        Close the pygame window.
        """
        pygame.quit()


# Test the visualizer if we run this file directly
if __name__ == "__main__":
    print("Testing Space Visualizer!")
    print("=" * 50)
    
    # Create visualizer
    viz = SpaceVisualizer(grid_size=5, fps=30)  # Uses default cell_size=100 for larger display
    
    # Simulate some steps
    import time
    
    for step in range(20):
        # Move robot in a simple pattern
        robot_pos = (step % 5, step // 5)
        
        # Create dummy environment info
        env_info = {
            'robot_pos': robot_pos,
            'asteroids': [(1, 1), (2, 3), (3, 1)],
            'energy_stars': [(1, 3), (3, 3)] if step < 10 else [(3, 3)],
            'goal_pos': (4, 4),
            'collected_energy': step // 5,
            'steps': step,
        }
        
        # Create dummy Q-values
        q_values = np.array([10, 5, 15, 8]) + np.random.randn(4) * 2
        
        # Reward changes over time
        reward = 10 if step in [5, 10, 15] else -1
        
        # Update display
        viz.set_episode(1)
        viz.set_epsilon(0.5)
        viz.update(env_info, reward, q_values, q_values, step, step % 4)
        
        # Check if we should quit
        if step == 0:
            print("Window created successfully!")
            print("Press ESC or close window to exit...")
    
    # Wait a bit before closing
    pygame.time.wait(2000)
    viz.close()
    
    print("Visualizer test complete!")
