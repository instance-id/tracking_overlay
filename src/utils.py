#!/usr/bin/env -S uv run

# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pygame>=2.5.0",
#   "numpy>=1.24.0"
# ]
# ///

"""
Utility functions for high-tech camera overlay system.
"""

import time
import pygame
import numpy as np
from datetime import datetime
from typing import Tuple, Union
import math
import cv2

MARGIN = 10  # pixels
ROW_SIZE = 30  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (0, 0, 0)  # black
DETECTION_DATA = []


def normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px


def write_detection_data():
  """Write detection data to a file."""
  with open('detection_data.txt', 'w') as f:
    split_text = ""

    # Split lines by comma
    for detection in DETECTION_DATA:
      split_text += f"{str(detection)}\n"

    f.write(str(split_text))


def visualize(
    image,
    detection_result,
    frame_width,
    frame_height,
) -> np.ndarray:
  """Draws bounding boxes on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualized.
    frame_width: The width of the input RGB image.
    frame_height: The height of the input RGB image.
  Returns:
    Image with bounding boxes.
  """
  for detection in detection_result.detections:

    category = detection.categories[0]
    if category.category_name != 'person':
      continue

    DETECTION_DATA.append(detection)

    # Draw bounding_box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    # Use the orange color for high visibility.
    cv2.rectangle(image, start_point, end_point, (0, 165, 255), 3)

    for keypoint in detection.keypoints:
      keypoint_px = normalized_to_pixel_coordinates(keypoint.x, keypoint.y, frame_width, frame_height)
      color, thickness, radius = (0, 255, 0), 2, 2
      cv2.circle(image, keypoint_px, thickness, color, radius)

    # Draw label and score
    category = detection.categories[0]
    category_name = (category.category_name if category.category_name is not None else '')
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (MARGIN + bbox.origin_x, MARGIN + ROW_SIZE + bbox.origin_y)
    cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_DUPLEX, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return image


def fps_counter(last_time, frame_count):
  """
  Calculate frames per second.
  
  Args:
      last_time: Last time measurement
      frame_count: Current frame count
      
  Returns:
      float: Current FPS
  """
  current_time = time.time()
  elapsed = current_time - last_time

  # Avoid division by zero
  if elapsed <= 0:
    return 0

  return frame_count / elapsed


def create_transparent_surface(width, height, color=(0, 0, 0, 0)):
  """
  Create transparent pygame surface.
  
  Args:
      width: Surface width
      height: Surface height
      color: RGBA color
      
  Returns:
      pygame.Surface: Transparent surface
  """
  surface = pygame.Surface((width, height), pygame.SRCALPHA)
  surface.fill(color)
  return surface


def draw_tech_text(surface, text, position, font_size=16, color=(0, 255, 255),
                   shadow=True, shadow_color=(0, 50, 100)):
  """
  Draw high-tech styled text with optional shadow.
  
  Args:
      surface: Pygame surface to draw on
      text: Text to draw
      position: (x, y) position
      font_size: Font size
      color: RGB color for text
      shadow: Whether to draw shadow
      shadow_color: RGB color for shadow
  """
  # Create font
  font = pygame.font.SysFont('monospace', font_size)

  # Draw shadow if requested
  if shadow:
    shadow_surface = font.render(text, True, shadow_color)
    shadow_pos = (position[0] + 2, position[1] + 2)
    surface.blit(shadow_surface, shadow_pos)

  # Draw main text
  text_surface = font.render(text, True, color)
  surface.blit(text_surface, position)


def draw_animated_border(surface, rect, color=(0, 255, 255), thickness=1,
                         animated=True, animation_speed=0.1):
  """
  Draw a high-tech animated border around a rectangle.
  
  Args:
      surface: Pygame surface to draw on
      rect: Pygame Rect object
      color: RGB color for border
      thickness: Border thickness
      animated: Whether to animate border
      animation_speed: Speed of animation
  """
  # Draw basic border
  pygame.draw.rect(surface, color, rect, thickness)

  if animated:
    # Get current time for animation
    t = pygame.time.get_ticks() * animation_speed * 0.001

    # Calculate animation phase for each corner
    phases = [
      t % 1.0,
      (t + 0.25) % 1.0,
      (t + 0.5) % 1.0,
      (t + 0.75) % 1.0
    ]

    # Corner coordinates
    corners = [
      (rect.left, rect.top),  # Top left
      (rect.right, rect.top),  # Top right
      (rect.right, rect.bottom),  # Bottom right
      (rect.left, rect.bottom)  # Bottom left
    ]

    # Draw animated corner markers
    for i, (corner, phase) in enumerate(zip(corners, phases)):
      # Calculate size based on phase
      size = int(5 + 3 * np.sin(phase * np.pi * 2))

      # Draw corner marker based on position
      if i == 0:  # Top left
        pygame.draw.line(surface, color, corner, (corner[0] + size, corner[1]), thickness)
        pygame.draw.line(surface, color, corner, (corner[0], corner[1] + size), thickness)
      elif i == 1:  # Top right
        pygame.draw.line(surface, color, corner, (corner[0] - size, corner[1]), thickness)
        pygame.draw.line(surface, color, corner, (corner[0], corner[1] + size), thickness)
      elif i == 2:  # Bottom right
        pygame.draw.line(surface, color, corner, (corner[0] - size, corner[1]), thickness)
        pygame.draw.line(surface, color, corner, (corner[0], corner[1] - size), thickness)
      else:  # Bottom left
        pygame.draw.line(surface, color, corner, (corner[0] + size, corner[1]), thickness)
        pygame.draw.line(surface, color, corner, (corner[0], corner[1] - size), thickness)


def timestamp_overlay(surface, position=(10, 10), font_size=16, color=(0, 255, 255)):
  """
  Draw timestamp overlay on surface.
  
  Args:
      surface: Pygame surface to draw on
      position: (x, y) position
      font_size: Font size
      color: RGB color for text
  """
  # Get current timestamp
  now = datetime.now()
  timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

  # Draw timestamp
  draw_tech_text(surface, timestamp, position, font_size, color)


def draw_data_box(surface, rect, title, data_dict, font_size=14, color=(0, 255, 255),
                  bg_color=(0, 0, 50, 150)):
  """
  Draw a high-tech data box with title and key-value pairs.
  
  Args:
      surface: Pygame surface to draw on
      rect: Pygame Rect for the box
      title: Box title
      data_dict: Dictionary of key-value pairs to display
      font_size: Font size
      color: RGB color for text
      bg_color: RGBA color for background
  """
  # Draw background
  box_surface = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
  box_surface.fill(bg_color)

  # Draw border
  draw_animated_border(box_surface, pygame.Rect(0, 0, rect.width, rect.height), color)

  # Draw title
  title_font = pygame.font.SysFont('monospace', font_size + 2, bold=True)
  title_surface = title_font.render(title, True, color)
  box_surface.blit(title_surface, (10, 5))

  # Draw horizontal line under title
  pygame.draw.line(
    box_surface,
    color,
    (5, 25),
    (rect.width - 5, 25),
    1
  )

  # Draw data
  y_offset = 35
  for key, value in data_dict.items():
    text = f"{key}: {value}"
    draw_tech_text(box_surface, text, (15, y_offset), font_size, color, shadow=False)
    y_offset += font_size + 6

  # Blit to main surface
  surface.blit(box_surface, (rect.left, rect.top))


class OverlayEffect:
  """Base class for animated overlay effects."""

  def __init__(self, duration=1.0):
    """
    Initialize overlay effect.
    
    Args:
        duration: Effect duration in seconds
    """
    self.duration = duration
    self.start_time = None
    self.active = False

  def start(self):
    """Start the effect."""
    self.start_time = time.time()
    self.active = True

  def update(self):
    """
    Update effect state.
    
    Returns:
        float: Progress from 0.0 to 1.0, or None if effect is completed
    """
    if not self.active:
      return None

    # Calculate elapsed time and progress
    elapsed = time.time() - self.start_time
    progress = elapsed / self.duration

    # Check if effect is completed
    if progress >= 1.0:
      self.active = False
      return None

    return progress

  def draw(self, surface):
    """
    Draw effect on surface.
    
    Args:
        surface: Pygame surface to draw on
        
    Returns:
        bool: True if effect is still active
    """
    progress = self.update()

    if progress is None:
      return False

    # Draw effect (to be implemented by subclasses)
    self._draw_effect(surface, progress)

    return True

  def _draw_effect(self, surface, progress):
    """
    Draw effect implementation.
    
    Args:
        surface: Pygame surface to draw on
        progress: Effect progress from 0.0 to 1.0
    """
    # To be implemented by subclasses
    pass


class ScanlineEffect(OverlayEffect):
  """Animated scanline effect."""

  def __init__(self, duration=2.0, color=(0, 255, 255), line_count=10):
    """
    Initialize scanline effect.
    
    Args:
        duration: Effect duration in seconds
        color: RGB color for scanlines
        line_count: Number of scanlines
    """
    super().__init__(duration)
    self.color = color
    self.line_count = line_count

  def _draw_effect(self, surface, progress):
    """Draw scanline effect."""
    width, height = surface.get_size()

    # Calculate base alpha based on progress
    if progress < 0.2:
      # Fade in
      base_alpha = int(255 * (progress / 0.2))
    elif progress > 0.8:
      # Fade out
      base_alpha = int(255 * (1 - (progress - 0.8) / 0.2))
    else:
      # Full alpha
      base_alpha = 255

    # Create transparent overlay
    overlay = pygame.Surface((width, height), pygame.SRCALPHA)

    # Draw scanlines
    for i in range(self.line_count):
      # Calculate y-position based on progress and line index
      phase = (progress * 2 + i / self.line_count) % 1.0
      y = int(height * phase)

      # Calculate line alpha
      line_alpha = base_alpha
      if phase < 0.1 or phase > 0.9:
        # Fade at edges
        edge_phase = min(phase, 1 - phase) * 10
        line_alpha = int(base_alpha * edge_phase)

      # Draw line
      color = (self.color[0], self.color[1], self.color[2], line_alpha)
      pygame.draw.line(overlay, color, (0, y), (width, y), 1)

    # Blit overlay to surface
    surface.blit(overlay, (0, 0))


class PulseEffect(OverlayEffect):
  """Circular pulse effect radiating from a point."""

  def __init__(self, center, duration=1.0, color=(255, 50, 50), max_radius=100):
    """
    Initialize pulse effect.
    
    Args:
        center: (x, y) center point
        duration: Effect duration in seconds
        color: RGB color for pulse
        max_radius: Maximum pulse radius
    """
    super().__init__(duration)
    self.center = center
    self.color = color
    self.max_radius = max_radius

  def _draw_effect(self, surface, progress):
    """Draw pulse effect."""
    # Calculate radius based on progress
    radius = int(self.max_radius * progress)

    # Calculate alpha based on progress (fade out as it expands)
    alpha = int(255 * (1 - progress))

    # Draw circle
    color = (self.color[0], self.color[1], self.color[2], alpha)
    pygame.draw.circle(surface, color, self.center, radius, 2)


class GlitchEffect(OverlayEffect):
  """Digital glitch effect."""

  def __init__(self, duration=0.5, intensity=0.2, rect=None):
    """
    Initialize glitch effect.
    
    Args:
        duration: Effect duration in seconds
        intensity: Glitch intensity (0.0 - 1.0)
        rect: Pygame Rect to affect, or None for entire surface
    """
    super().__init__(duration)
    self.intensity = intensity
    self.rect = rect
    self.glitch_blocks = []

  def start(self):
    """Start the effect and generate glitch blocks."""
    super().start()

    # Generate random glitch blocks
    self.glitch_blocks = []

    # Number of blocks depends on intensity
    num_blocks = int(5 + self.intensity * 15)

    for _ in range(num_blocks):
      # Random block parameters
      block = {
        'x_offset': np.random.randint(-10, 10),
        'y_offset': np.random.randint(-10, 10),
        'width': np.random.randint(20, 100),
        'height': np.random.randint(5, 20),
        'x': np.random.randint(0, 100),
        'y': np.random.randint(0, 100),
        'color_shift': np.random.randint(0, 50),
        'alpha': np.random.randint(50, 200)
      }

      self.glitch_blocks.append(block)

  def _draw_effect(self, surface, progress):
    """Draw glitch effect."""
    # Get surface dimensions
    width, height = surface.get_size()

    # Determine affect area
    if self.rect:
      x, y, w, h = self.rect.x, self.rect.y, self.rect.width, self.rect.height
    else:
      x, y, w, h = 0, 0, width, height

    # Create temporary surface for the affected area
    temp = surface.subsurface(pygame.Rect(x, y, w, h)).copy()

    # Apply glitch blocks
    for block in self.glitch_blocks:
      # Calculate block position as percentage of effect area
      block_x = x + int(w * block['x'] / 100)
      block_y = y + int(h * block['y'] / 100)

      # Skip if block is outside surface
      if block_x >= width or block_y >= height:
        continue

      # Adjust block size to fit within surface
      block_w = min(block['width'], width - block_x)
      block_h = min(block['height'], height - block_y)

      if block_w <= 0 or block_h <= 0:
        continue

      # Create copy of block region
      try:
        block_surface = surface.subsurface(
          pygame.Rect(block_x, block_y, block_w, block_h)
        ).copy()

        # Apply color shift
        if block['color_shift'] > 0:
          # Convert to array for manipulation
          pixels = pygame.surfarray.pixels3d(block_surface)

          # Shift a random color channel
          channel = np.random.randint(0, 3)
          pixels[:, :, channel] = np.clip(
            pixels[:, :, channel] + block['color_shift'],
            0,
            255
          )

          # Release array
          del pixels

        # Draw shifted block
        surface.blit(
          block_surface,
          (block_x + block['x_offset'], block_y + block['y_offset']),
          special_flags=pygame.BLEND_RGBA_ADD
        )

      except ValueError:
        # Skip if subsurface is invalid
        continue

    # Apply scanlines based on progress
    scanline_count = int(10 * self.intensity)
    for i in range(scanline_count):
      # Random scanline parameters
      sy = y + int(h * np.random.random())
      sh = np.random.randint(1, 3)
      alpha = np.random.randint(100, 200)

      # Draw scanline
      pygame.draw.rect(
        surface,
        (200, 200, 200, alpha),
        pygame.Rect(x, sy, w, sh)
      )

    # Apply overall glitchy noise based on progress
    if progress < 0.2 or progress > 0.8 or np.random.random() < 0.3:
      noise_alpha = int(100 * self.intensity * (1 - abs(progress - 0.5) * 2))
      noise_surface = pygame.Surface((w, h), pygame.SRCALPHA)

      # Generate noise
      for nx in range(0, w, 2):
        for ny in range(0, h, 2):
          if np.random.random() < 0.1 * self.intensity:
            noise_color = (
              np.random.randint(0, 255),
              np.random.randint(0, 255),
              np.random.randint(0, 255),
              noise_alpha
            )
            pygame.draw.rect(
              noise_surface,
              noise_color,
              pygame.Rect(nx, ny, 2, 2)
            )

      # Apply noise
      surface.blit(noise_surface, (x, y), special_flags=pygame.BLEND_RGBA_ADD)


# Effect manager for coordinating multiple overlay effects
class EffectManager:
  """Manager for coordinating multiple overlay effects."""

  def __init__(self):
    """Initialize effect manager."""
    self.effects = []

  def add_effect(self, effect):
    """
    Add effect to manager.
    
    Args:
        effect: OverlayEffect instance
    """
    effect.start()
    self.effects.append(effect)

  def update_and_draw(self, surface):
    """
    Update and draw all active effects.
    
    Args:
        surface: Pygame surface to draw on
    """
    # Update and draw effects, removing completed ones
    active_effects = []

    for effect in self.effects:
      if effect.draw(surface):
        active_effects.append(effect)

    # Update active effects list
    self.effects = active_effects

  def clear(self):
    """Clear all effects."""
    self.effects = []


if __name__ == "__main__":
  # Simple test code
  import time

  # Initialize pygame
  pygame.init()

  # Create display
  width, height = 800, 600
  screen = pygame.display.set_mode((width, height))
  pygame.display.set_caption("Effects Test")

  # Create effect manager
  manager = EffectManager()

  # Main loop
  running = True
  while running:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        running = False
      elif event.type == pygame.MOUSEBUTTONDOWN:
        # Add random effect on click
        effect_type = np.random.randint(0, 3)
        pos = pygame.mouse.get_pos()

        if effect_type == 0:
          manager.add_effect(ScanlineEffect())
        elif effect_type == 1:
          manager.add_effect(PulseEffect(pos))
        else:
          manager.add_effect(GlitchEffect(rect=pygame.Rect(pos[0] - 50, pos[1] - 50, 100, 100)))

    # Clear screen
    screen.fill((0, 0, 40))

    # Draw timestamp
    timestamp_overlay(screen)

    # Draw data box
    data = {
      "FPS": "60.0",
      "Status": "Active",
      "Objects": "5",
      "Memory": "128 MB"
    }
    draw_data_box(screen, pygame.Rect(50, 50, 200, 150), "System Status", data)

    # Update and draw effects
    manager.update_and_draw(screen)

    # Update display
    pygame.display.flip()

    # Cap frame rate
    pygame.time.Clock().tick(60)

  pygame.quit()
