#!/usr/bin/env -S uv run

# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pygame>=2.5.0",
#   "numpy>=1.24.0"
# ]
# ///

"""
Grid overlay module for high-tech camera overlay system.
Handles rendering and animation of the grid overlay.
"""

import pygame
import numpy as np


class Grid:
  """Grid overlay class for creating a high-tech grid visualization."""

  def __init__(self, cols=9, rows=5, width=1280, height=720,
               line_color=(0, 255, 255, 150), line_thickness=1, animation_speed=0.1,
               cell_highlight_alpha=50, cell_active_color=(255, 200, 50), cell_inactive_color=(50, 200, 255)):
    """
    Initialize grid overlay.

    Args:
        cols: Number of grid columns
        rows: Number of grid rows
        width: Display width
        height: Display height
        line_color: RGBA color for grid lines
        line_thickness: Thickness of grid lines
        animation_speed: Speed of grid animations
        cell_highlight_alpha: Alpha for cell highlights
        cell_active_color: Color for active cells
        cell_inactive_color: Color for inactive cells
    """
    self.cols = cols
    self.rows = rows
    self.width = width
    self.height = height
    self.line_color = line_color
    self.line_thickness = line_thickness
    self.animation_speed = animation_speed

    # Grid cell settings
    self.cell_highlight_alpha = cell_highlight_alpha
    self.cell_active_color = cell_active_color
    self.cell_inactive_color = cell_inactive_color

    # Calculate cell dimensions
    self.cell_width = width // cols
    self.cell_height = height // rows

    # Initialize cell states (for animation and highlighting)
    self.cell_states = np.zeros((rows, cols, 4), dtype=np.float32)
    # Format: [highlight_alpha, animation_phase, highlight_duration, is_active]

    # Create surfaces
    self.setup_surfaces()

  def setup_surfaces(self):
    """Create necessary surfaces for rendering."""
    # Main grid surface (transparent)
    self.grid_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

    # Cell highlight surface (transparent, for animated highlights)
    self.highlight_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

  def update(self, active_cells=None):
    """
    Update grid state with active cells.

    Args:
        active_cells: 2D numpy array of shape (rows, cols) with 1 for active cells, 0 for inactive
    """
    # If no active cells provided, gradually decay all cells
    if active_cells is None:
      # Reduce highlight alpha
      self.cell_states[:, :, 0] = np.maximum(0, self.cell_states[:, :, 0] - self.animation_speed)
      # Advance animation phase for active cells
      mask = self.cell_states[:, :, 3] > 0
      self.cell_states[mask, 1] = (self.cell_states[mask, 1] + self.animation_speed) % 1.0
      # Reduce highlight duration
      self.cell_states[:, :, 2] = np.maximum(0, self.cell_states[:, :, 2] - self.animation_speed)
      # Deactivate cells with zero duration
      self.cell_states[self.cell_states[:, :, 2] <= 0, 3] = 0
    else:
      # Ensure correct shape
      if active_cells.shape != (self.rows, self.cols):
        raise ValueError(f"active_cells shape {active_cells.shape} does not match grid shape {(self.rows, self.cols)}")

      # Update cell states based on active cells
      for r in range(self.rows):
        for c in range(self.cols):
          if active_cells[r, c] > 0:
            # Activate cell
            self.cell_states[r, c, 0] = 1.0  # Full highlight
            self.cell_states[r, c, 1] = 0.0  # Reset animation phase
            self.cell_states[r, c, 2] = 1.0  # Set highlight duration
            self.cell_states[r, c, 3] = 1.0  # Mark as active

      # Also decay inactive cells
      inactive_mask = active_cells == 0
      self.cell_states[inactive_mask, 0] = np.maximum(0, self.cell_states[inactive_mask, 0] - self.animation_speed)
      self.cell_states[inactive_mask, 2] = np.maximum(0, self.cell_states[inactive_mask, 2] - self.animation_speed)
      self.cell_states[self.cell_states[:, :, 2] <= 0, 3] = 0

  def draw(self, surface):
    """
    Draw grid overlay on the provided surface.

    Args:
        surface: Pygame surface to draw on
    """
    # Clear surfaces
    self.grid_surface.fill((0, 0, 0, 0))
    self.highlight_surface.fill((0, 0, 0, 0))

    # Draw static grid lines
    self._draw_grid_lines()

    # Draw cell highlights
    self._draw_cell_highlights()

    # Blit grid and highlights to main surface
    surface.blit(self.highlight_surface, (0, 0))
    surface.blit(self.grid_surface, (0, 0))

  def _draw_grid_lines(self):
    """Draw the static grid lines."""
    # Set alpha for line color
    color = self.line_color

    # Draw vertical lines
    for i in range(self.cols + 1):
      x = i * self.cell_width
      pygame.draw.line(
        self.grid_surface,
        color,
        (x, 0),
        (x, self.height),
        self.line_thickness
      )

    # Draw horizontal lines
    for j in range(self.rows + 1):
      y = j * self.cell_height
      pygame.draw.line(
        self.grid_surface,
        color,
        (0, y),
        (self.width, y),
        self.line_thickness
      )

  def _draw_cell_highlights(self):
    """Draw the animated cell highlights."""
    # Process each cell
    for r in range(self.rows):
      for c in range(self.cols):
        # Skip cells with no highlight
        if self.cell_states[r, c, 0] <= 0:
          continue

        # Calculate cell coordinates
        x = c * self.cell_width
        y = r * self.cell_height

        # Get highlight alpha and color
        alpha = int(self.cell_states[r, c, 0] * self.cell_highlight_alpha)  # Max alpha 150

        # Create highlight color based on cell state
        is_active = self.cell_states[r, c, 3] > 0
        if is_active:
          # Pulse animation for active cells
          phase = self.cell_states[r, c, 1]
          pulse_alpha = int(np.sin(phase * np.pi) * 50) + 50
          alpha = min(255, alpha + pulse_alpha)

          # Active cell color (yellow/orange)
          color = (self.cell_active_color[0], self.cell_active_color[1], self.cell_active_color[2], alpha)
        else:
          # Inactive cell color (blue/cyan)
          color = (self.cell_inactive_color[0], self.cell_inactive_color[1], self.cell_inactive_color[2], alpha)

        # Draw cell rectangle
        rect = pygame.Rect(x, y, self.cell_width, self.cell_height)
        pygame.draw.rect(self.highlight_surface, color, rect)

        # Draw cell border with higher alpha
        border_color = (color[0], color[1], color[2], min(255, color[3] + 50))
        pygame.draw.rect(
          self.highlight_surface,
          border_color,
          rect,
          self.line_thickness
        )

  def get_cell_at_position(self, pos):
    """
    Get grid cell indices at the given position.

    Args:
        pos: (x, y) position in pixels

    Returns:
        Tuple (row, col) of cell indices, or None if outside grid
    """
    x, y = pos

    # Check if position is within grid
    if x < 0 or x >= self.width or y < 0 or y >= self.height:
      return None

    # Calculate cell indices
    col = int(x // self.cell_width)
    row = int(y // self.cell_height)

    # Clamp to valid range
    col = max(0, min(col, self.cols - 1))
    row = max(0, min(row, self.rows - 1))

    return (row, col)

  def highlight_cell(self, row, col, duration=1.0):
    """
    Manually highlight a specific cell.

    Args:
        row: Cell row index
        col: Cell column index
        duration: Duration of highlight in seconds
    """
    if 0 <= row < self.rows and 0 <= col < self.cols:
      self.cell_states[row, col, 0] = 1.0  # Full highlight
      self.cell_states[row, col, 1] = 0.0  # Reset animation phase
      self.cell_states[row, col, 2] = duration  # Set highlight duration
      self.cell_states[row, col, 3] = 1.0  # Mark as active


if __name__ == "__main__":
  # Simple test code
  import time

  # Initialize pygame
  pygame.init()

  # Create display
  width, height = 1280, 720
  screen = pygame.display.set_mode((width, height))
  pygame.display.set_caption("Grid Test")

  # Create grid
  grid = Grid(9, 5, width, height)

  # Main loop
  running = True
  previous_cell = None
  while running:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        running = False
      elif event.type == pygame.MOUSEMOTION or event.type == pygame.MOUSEBUTTONDOWN or event.type == pygame.MOUSEBUTTONUP or event.type == pygame.MOUSEWHEEL:
        # Highlight cell on click
        pos = pygame.mouse.get_pos()
        cell = grid.get_cell_at_position(pos)

        if cell is not None and cell != previous_cell:
          grid.highlight_cell(*cell)
          previous_cell = cell

    # Clear screen
    screen.fill((0, 0, 0))

    # Update and draw grid
    grid.update()
    grid.draw(screen)

    # Update display
    pygame.display.flip()

    # Cap frame rate
    pygame.time.Clock().tick(30)

  pygame.quit()
