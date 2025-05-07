#!/usr/bin/env -S uv run

# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pygame>=2.5.0",
#   "numpy>=1.24.0",
#   "pydantic>=2.0.0"
# ]
# ///

"""
Reticle module for high-tech camera overlay system.
Handles rendering and animation of head tracking reticle.

Simple, direct implementation for guaranteed rotation.
"""

import math
import pygame
import numpy as np
from typing import Optional, Tuple, Dict, Any, List, Union
import time
from time import time


class Reticle:
  """Animated reticle for tracking and highlighting detected objects.
  
  This is a simple, direct implementation that guarantees visible rotation.
  """

  def __init__(self, size=50, color=(255, 50, 50), line_thickness=1, animation_speed=0.1,
               smoothing_factor=0.15, min_speed=0.5, max_speed=30.0, dampening_factor=0.8,
               lerp_factor=0.2):
    """
    Initialize reticle with direct rotation implementation.

    Args:
        size: Size of reticle in pixels
        color: RGB color tuple
        line_thickness: Thickness of lines
        animation_speed: Speed of animations
        smoothing_factor: Position smoothing factor (lower = smoother but slower)
        min_speed: Minimum movement speed in pixels
        max_speed: Maximum movement speed in pixels
        dampening_factor: Dampening factor for micro-movements
        lerp_factor: Linear interpolation factor (lower = smoother transitions, range 0.01-1.0)
    """
    # Store parameters
    self.size = size
    if isinstance(color, tuple):
      if len(color) == 3:
        self.color = (color[0], color[1], color[2], 255)  # Add alpha if needed
      else:
        self.color = color
    else:
      self.color = (255, 50, 50, 155)  # Default color with alpha

    self.line_thickness = line_thickness
    self.animation_speed = animation_speed
    self.smoothing_factor = smoothing_factor
    self.min_speed = min_speed
    self.max_speed = max_speed
    self.dampening_factor = dampening_factor
    self.lerp_factor = lerp_factor

    # Position tracking
    self.rotation = 0
    self.scale = 1.0
    self.active = False
    self.target_position = None
    self.current_position = None
    self.lerp_target = None
    self.velocity = (0.0, 0.0)
    self.last_positions = []
    self.max_position_history = 5

    # DIRECT ROTATION ELEMENTS
    # Create two separate rotation angles for different elements
    self.outer_rotation = 0.0
    self.inner_rotation = 0.0

    # Set rotation speeds - make them much higher than default to ensure visibility
    self.outer_rotation_speed = 1.5  # radians per second
    self.inner_rotation_speed = -1.1  # negative for counter-clockwise

    # For backward compatibility with overlay_app.py
    # This structure is used by the app, so we need to maintain compatibility
    self._legacy_rotating_elements = [
      {'angle': 0.0, 'speed': animation_speed, 'scale': 0.9, 'direction': 1, 'active': True,
       'color': color, 'thickness': line_thickness, 'segment_arc': math.pi / 4, 'offset': 1.57},
      {'angle': math.pi / 2, 'speed': animation_speed, 'scale': 1.0, 'direction': 1, 'active': True,
       'color': color, 'thickness': line_thickness, 'segment_arc': math.pi / 2, 'offset': 1.57},
      {'angle': math.pi, 'speed': animation_speed, 'scale': 0.9, 'direction': 1, 'active': True,
       'color': color, 'thickness': line_thickness, 'segment_arc': math.pi / 4, 'offset': 1.57},
      {'angle': 3 * math.pi / 2, 'speed': animation_speed, 'scale': 1.0, 'direction': 1, 'active': True,
       'color': color, 'thickness': line_thickness, 'segment_arc': math.pi / 2, 'offset': 1.57}
    ]

    # Create surface with extra margin
    surface_size = int(size * 3)
    self.surface = pygame.Surface((surface_size, surface_size), pygame.SRCALPHA)

    # Blue color for the second ring
    self.blue_color = (50, 150, 255, 255)
    self.yellow_color = (255, 255, 50, 200)
    self.inner_color =(255, 50, 55, 220)
    self.outer_color = (255, 255, 50, 200)

    # Time tracking
    self.last_update_time = time()
    
    # Animation state
    self._animation_active = True

  def update(self, target_position=None):
    """
    Update reticle animation state with direct rotation implementation.

    Args:
        target_position: (x, y) position to track, or None to deactivate
    """
    # Calculate time delta for animation
    current_time = time()
    dt = current_time - self.last_update_time
    self.last_update_time = current_time

    # DIRECT ROTATION CALCULATION
    # Update rotation angles explicitly with real-time
    self.outer_rotation += self.outer_rotation_speed * dt
    self.inner_rotation += self.inner_rotation_speed * dt

    # Keep angles in 0-2π range
    self.outer_rotation %= math.pi * 2
    self.inner_rotation %= math.pi * 2

    # Position tracking
    if target_position:
      self.active = True
      self._animation_active = True  # Ensure animation is always active when we have a target
  
      # Apply position smoothing and tracking
      if not self.current_position:
        # First position, just set it directly
        self.current_position = target_position
        # Also set the target_position attribute for compatibility
        self.target_position = target_position
      else:
        # Store the current position for dampening
        if len(self.last_positions) >= self.max_position_history:
          self.last_positions.pop(0)
        self.last_positions.append(target_position)
  
        # Apply smoothing
        target_x, target_y = target_position
        current_x, current_y = self.current_position
  
        # Calculate distance to target
        dx = target_x - current_x
        dy = target_y - current_y
        distance = math.sqrt(dx * dx + dy * dy)
  
        # Determine speed based on distance
        speed = max(self.min_speed, min(distance * self.smoothing_factor, self.max_speed))
  
        # Apply movement with lerp factor
        if distance > 0.1:  # Only move if we're not already very close
          self.current_position = self.lerp(self.current_position, target_position, self.lerp_factor)
        
        # Update the actual target_position property
        self.target_position = target_position
    else:
      self.active = False
      self._animation_active = False

    # Return current position for convenience
    return self.current_position

  def draw(self, surface, position=None):
    """
    Draw reticle with direct rotation implementation.

    Args:
        surface: Pygame surface to draw on
        position: (x, y) position override, or None to use tracked position
    """
    # Clear the surface
    self.surface.fill((0, 0, 0, 0))

    # Get the position to draw at
    pos = position or self.current_position
    if not pos:
      return
      
    # If position is provided, ensure target position is updated
    if position is not None:
      self.target_position = position
      self.active = True

    # Get the center of our drawing surface
    center = (self.surface.get_width() // 2, self.surface.get_height() // 2)

    # DIRECT DRAWING IMPLEMENTATION
    # Draw the static center dot
    center_dot_color = (self.color[0],self.color[1],self.color[2], 175)
    # pygame.draw.circle(self.surface, center_dot_color, center, self.size * 0.05)

    # Draw the rotating outer ring (RED)
    self._draw_rotating_arc_segments(
      self.surface, center,
      self.size * 0.9, self.size * 0.89,
      self.outer_color,
      self.outer_rotation,
      3
    )

    # Draw the rotating inner ring (BLUE)
    self._draw_rotating_arc_segments(
      self.surface, center,
      self.size * 0.87, self.size * 0.86,
      self.inner_color,
      self.inner_rotation,
      4
    )

    # Blit to target surface
    dest_x = int(pos[0] - self.surface.get_width() // 2)
    dest_y = int(pos[1] - self.surface.get_height() // 2)
    surface.blit(self.surface, (dest_x, dest_y))

  def _draw_rotating_arc_segments(self, surface, center, outer_radius, inner_radius, color, base_rotation, num_segments):
    """
    Draw rotating arc segments around a circle with consistent line thickness.
    
    Args:
        surface: Surface to draw on
        center: Center position (x, y)
        outer_radius: Outer radius of the arc segments
        inner_radius: Inner radius of the arc segments
        color: Color to draw the segments with
        base_rotation: Base rotation angle in radians
        num_segments: Number of segments to draw
    """
    # Calculate angle per segment
    angle_per_segment = math.pi * 2 / num_segments

    # Calculate arc width (with some space between segments)
    arc_width = angle_per_segment * 0.7  # 70% of the segment space to leave gaps

    # Use a slightly thinner line thickness for more consistent rendering
    thickness = max(1, self.line_thickness - 1)

    # Draw each segment
    for i in range(num_segments):
      # Calculate segment start and end angles
      segment_angle = i * angle_per_segment + base_rotation
      start_angle = segment_angle - arc_width / 2
      end_angle = segment_angle + arc_width / 2

      # Calculate points for each segment to draw lines instead of arcs
      # This gives more consistent thickness than pygame.draw.arc
      num_points = 24  # More points = smoother curve but more expensive
      outer_points = []
      inner_points = []

      for j in range(num_points + 1):
        # Calculate the angle for this point
        angle = start_angle + (end_angle - start_angle) * (j / num_points)

        # Calculate outer and inner points
        outer_point = (
          center[0] + math.cos(angle) * outer_radius,
          center[1] + math.sin(angle) * outer_radius
        )
        inner_point = (
          center[0] + math.cos(angle) * inner_radius,
          center[1] + math.sin(angle) * inner_radius
        )

        outer_points.append(outer_point)
        inner_points.append(inner_point)

      # Draw the outer arc as connected lines
      for j in range(len(outer_points) - 1):
        pygame.draw.line(surface, color, outer_points[j], outer_points[j + 1], thickness)
      # 
      # Draw the inner arc as connected lines
      for j in range(len(inner_points) - 1):
        pygame.draw.line(surface, color, inner_points[j], inner_points[j + 1], thickness)

      # Draw the connecting lines at start and end
      pygame.draw.line(surface, color, outer_points[0], inner_points[0], thickness)
      pygame.draw.line(surface, color, outer_points[-1], inner_points[-1], thickness)

  def lerp(self, start, end, factor):
    """
    Linear interpolation between two points.

    Args:
        start: Starting point as (x, y) tuple
        end: Ending point as (x, y) tuple
        factor: Interpolation factor between 0 and 1

    Returns:
        Interpolated position as (x, y) tuple
    """
    return (
      start[0] + (end[0] - start[0]) * factor,
      start[1] + (end[1] - start[1]) * factor
    )

  def set_element_speed(self, index, speed):
    """
    Set the rotation speed of a specific rotating element.

    Args:
        index: Index of the element (0-3)
        speed: New rotation speed value
    """
    if index == 0:
      self.outer_rotation_speed = speed
    elif index == 1:
      self.inner_rotation_speed = -speed  # Make inner ring rotate opposite direction

  def set_element_direction(self, index, clockwise=True):
    """
    Set the rotation direction of a specific rotating element.

    Args:
        index: Index of the element (0-3)
        clockwise: True for clockwise, False for counter-clockwise
    """
    direction = 1 if clockwise else -1
    if index == 0:
      self.outer_rotation_speed = abs(self.outer_rotation_speed) * direction
    elif index == 1:
      self.inner_rotation_speed = abs(self.inner_rotation_speed) * -direction  # Opposite of outer

  def set_element_properties(self, index, **properties):
    """
    Set multiple properties of a rotating element at once.

    Args:
        index: Index of the element (0-3)
        **properties: Any element properties to change (speed, direction, active, color, etc.)
    """
    if 'speed' in properties:
      self.set_element_speed(index, properties['speed'])
    if 'direction' in properties:
      clockwise = properties['direction'] > 0
      self.set_element_direction(index, clockwise)

  def set_all_speeds(self, speed):
    """Set the speed of all rotating elements."""
    self.outer_rotation_speed = speed
    self.inner_rotation_speed = -speed  # Opposite direction

  def set_all_directions(self, clockwise=True):
    """Set the direction of all rotating elements."""
    direction = 1 if clockwise else -1
    self.outer_rotation_speed = abs(self.outer_rotation_speed) * direction
    self.inner_rotation_speed = abs(self.inner_rotation_speed) * -direction  # Opposite of outer

  @property
  def rotating_elements(self):
    """Property getter for backward compatibility with overlay_app.py.
    Returns the legacy rotating elements structure."""
    return self._legacy_rotating_elements


if __name__ == "__main__":
  # Simple test code
  from time import time

  # Initialize pygame
  pygame.init()

  # Create display
  width, height = 800, 600
  screen = pygame.display.set_mode((width, height))
  pygame.display.set_caption("Reticle Test")

  # Create reticle
  reticle = Reticle(75, (255, 50, 50, 175), 1)

  # Main loop
  running = True
  clock = pygame.time.Clock()
  while running:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        running = False

    # Clear screen
    screen.fill((0, 0, 0))

    # Update reticle position to mouse
    mouse_pos = pygame.mouse.get_pos()
    reticle.update(mouse_pos)

    # Draw reticle
    reticle.draw(screen)

    # Update display
    pygame.display.flip()

    # Cap frame rate
    clock.tick(60)

  pygame.quit()
