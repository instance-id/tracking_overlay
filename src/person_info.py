#!/usr/bin/env -S uv run

# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pygame>=2.5.0",
#   "numpy>=1.24.0"
# ]
# ///

"""
Person information panel module for high-tech camera overlay system.
Handles rendering and animation of person identification displays.
"""

import math
import pygame
import numpy as np
from datetime import datetime


class PersonInfoPanel:
  """Display panel for showing information about identified persons."""

  def __init__(
      self,
      width=120,
      height=60,
      header_color=(25, 25, 25, 100),
      color=(10, 10, 50, 200),
      text_color=(0, 255, 255),
      font_size=10,
  ):
    """
    Initialize person information panel.

    Args:
        width: Panel width in pixels
        height: Panel height in pixels
        color: RGBA color tuple for panel background
        text_color: RGB color tuple for text
        font_size: Font size for text
    """
    self.header_color = header_color
    self.width = width
    self.height = height
    self.color = color
    self.text_color = text_color
    self.font_size = font_size
    self.base_height_pos = 30

    # Content
    self.person_id = None
    self.name = "Unknown"
    self.status = "Scanning..."
    self.confidence = 0.0
    self.timestamp = datetime.now()

    # Animation state
    self.alpha = 0
    self.target_alpha = 0
    self.animation_progress = 0
    self.active = False

    # Initialize pygame font
    pygame.font.init()
    self.font = pygame.font.SysFont('monospace', font_size, bold=True)
    self.title_font = pygame.font.SysFont('monospace', font_size + 1, bold=True)

    # Create surface
    self.surface = pygame.Surface((width, height), pygame.SRCALPHA)

  def update(self, person_id=None, name=None, status=None, confidence=None):
    """
    Update panel content.

    Args:
        person_id: ID for the person
        name: Name of the person
        status: Status text
        confidence: Confidence value (0.0 - 1.0)
    """
    # Update content if provided
    if person_id is not None:
      self.person_id = person_id

    if name is not None:
      self.name = name

    if status is not None:
      self.status = status

    if confidence is not None:
      self.confidence = confidence

    # Update timestamp
    self.timestamp = datetime.now()

    # Activate panel
    self.active = True
    self.target_alpha = 255

  def is_valid_color(self, color):
    """
    Check if the color is valid (not transparent).

    Args:
        color: RGBA color tuple

    Returns:
        bool: True if valid, False otherwise
    """
    return color[3] > 0

  def draw(self, surface, position):
    """
    Draw panel at the specified position.

    Args:
        surface: Pygame surface to draw on
        position: (x, y) position for top-left corner
    """
    # Update animation
    self._update_animation()

    # Skip if not visible
    if self.alpha <= 0:
      return

    # Clear panel surface
    self.surface.fill((0, 0, 0, 0))

    # Draw background with current alpha
    
    # Is Valid Color
    if self.color[3] > 0:
      bg_color = (self.color[0], self.color[1], self.color[2], min(self.color[3], min(220, self.alpha)))
    else:
      bg_color = (self.color[0], self.color[1], self.color[2], min(220, self.alpha))
    
    pygame.draw.rect(
      self.surface,
      bg_color,
      (0, 0, self.width, self.height),
      0,  # Filled
      border_radius=5
    )

    # Draw border
    border_color = (self.text_color[0], self.text_color[1], self.text_color[2], min(200, self.alpha))
    # pygame.draw.rect(
    #   self.surface,
    #   border_color,
    #   (0, 0, self.width, self.height),
    #   0,  # Border only
    #   border_radius=2
    # )

    # Draw header
    header_rect = pygame.Rect(0, 0, self.width, 15)
    header_color = (self.header_color[0], self.header_color[1], self.header_color[2], min(220, self.alpha))
    pygame.draw.rect(
      self.surface,
      header_color,
      header_rect,
      0,  # Filled
      border_radius=2
    )

    # Draw top rounded corners separately to avoid overlap
    pygame.draw.rect(
      self.surface,
      header_color,
      (0, 5, self.width, 15),
      0  # Filled
    )

    # Draw content with animation
    self._draw_content()

    # Draw to main surface
    x, y = position
    surface.blit(self.surface, (x, y))

  def _update_animation(self):
    """Update animation state."""
    # Smooth alpha transition
    if self.active:
      self.alpha += (self.target_alpha - self.alpha) * 0.2
      if self.alpha >= self.target_alpha - 5:
        self.alpha = self.target_alpha

      # Increase animation progress
      self.animation_progress = min(1.0, self.animation_progress + 0.05)
    else:
      self.alpha -= (self.alpha - 0) * 0.2
      if self.alpha <= 5:
        self.alpha = 0

      # Decrease animation progress
      self.animation_progress = max(0.0, self.animation_progress - 0.05)

    # Deactivate if no update for 5 seconds
      # Removing this auto-deactivation as it can cause intermittent disappearance
      # when tracking is actually still active but the panel wasn't explicitly updated
      # time_diff = (datetime.now() - self.timestamp).total_seconds()
      # if time_diff > 5:
      #   self.active = False

  def _get_height_pos(self, height_pos):
    """Get height position for a given height and font size."""
    self.base_height_pos = height_pos + int(self.font_size)
    return self.base_height_pos

  def _draw_content(self):
    """Draw panel content with animation effects."""
    # Calculate text alpha
    text_alpha = min(255, self.alpha)
    text_color = (self.text_color[0], self.text_color[1], self.text_color[2], text_alpha)

    # Draw title
    # pad spacing after ID: so that person_id is right-aligned with name and status

    title = "ID: "
    if self.person_id is not None:
      title += f"               {self.person_id}"
    else:
      title += "      Scanning..."

    title_surface = self.title_font.render(title, True, text_color)
    self.surface.blit(title_surface, (10, 5))

    # Calculate animation offset for each line
    anim_offset = (1.0 - self.animation_progress) * 50
    self.base_height_pos = 25

    # Draw name with animation
    name_surface = self.font.render(f"Name:      {self.name}", True, text_color)
    self.surface.blit(name_surface, (15, self.base_height_pos + anim_offset * 0.2))

    # Draw status with animation
    status_surface = self.font.render(f"Status:      {self.status}", True, text_color)
    self.surface.blit(status_surface, (15, self._get_height_pos(self.base_height_pos) + anim_offset * 0.4))

    # Draw confidence with animation
    conf_text = f"Confidence:   {self.confidence:.1%}"
    conf_surface = self.font.render(conf_text, True, text_color)
    self.surface.blit(conf_surface, (15, self._get_height_pos(self.base_height_pos) + anim_offset * 0.6))

    # Draw timestamp with animation
    time_text = f"Time:      {self.timestamp.strftime('%H:%M:%S')}"
    time_surface = self.font.render(time_text, True, text_color)
    self.surface.blit(time_surface, (15, self._get_height_pos(self.base_height_pos) + anim_offset * 0.8))

    # Draw animated scanning line
    # self._draw_scanning_effect()

  def _draw_scanning_effect(self):
    """Draw animated scanning effect."""
    # Only draw if active
    if not self.active:
      return

    # Calculate scan line position (oscillates)
    scan_pos = int((math.sin(pygame.time.get_ticks() * 0.005) + 1) * 0.5 * self.height)

    # Draw horizontal scan line
    scan_color = (self.text_color[0], self.text_color[1], self.text_color[2], 50)
    pygame.draw.line(
      self.surface,
      scan_color,
      (0, scan_pos),
      (self.width, scan_pos),
      1
    )

    # Draw vertical pulse lines
    pulse_height = 10
    alpha = 50 + int(20 * math.sin(pygame.time.get_ticks() * 0.01))
    pulse_color = (self.text_color[0], self.text_color[1], self.text_color[2], alpha)

    # Draw left and right pulse indicators
    for i in range(pulse_height):
      # Fade alpha based on distance from scan line
      fade_alpha = alpha * (pulse_height - i) / pulse_height
      pulse_color = (self.text_color[0], self.text_color[1], self.text_color[2], int(fade_alpha))

      # Draw pulse lines
      pygame.draw.line(
        self.surface,
        pulse_color,
        (0, scan_pos - i),
        (3, scan_pos - i),
        1
      )
      pygame.draw.line(
        self.surface,
        pulse_color,
        (self.width - 4, scan_pos - i),
        (self.width, scan_pos - i),
        1
      )


if __name__ == "__main__":
  # Simple test code
  import time

  # Initialize pygame
  pygame.init()

  # Create display
  width, height = 800, 600
  screen = pygame.display.set_mode((width, height))
  pygame.display.set_caption("Person Info Panel Test")

  # Create panel
  panel = PersonInfoPanel(160, 80)

  # Update with some data
  panel.update(
    person_id="P-1201",
    name="John Doe",
    status="Authorized",
    confidence=0.92
  )

  # Main loop
  running = True
  while running:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        running = False
      elif event.type == pygame.MOUSEBUTTONDOWN:
        # Update panel on click
        panel.update(
          person_id="P-" + str(np.random.randint(1000, 9999)),
          name=np.random.choice(["John Doe", "Jane Smith", "Alex Johnson"]),
          status=np.random.choice(["Authorized", "Visitor", "Unknown", "Restricted"]),
          confidence=np.random.random()
        )

    # Clear screen
    screen.fill((0, 0, 0))

    # Draw panel
    panel.draw(screen, (50, 50))

    # Update display
    pygame.display.flip()

    # Cap frame rate
    pygame.time.Clock().tick(60)

  pygame.quit()
