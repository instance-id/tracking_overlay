#!/usr/bin/env -S uv run

# /// script
# requires-python = ">=3.10"
# dependencies = [
#  "pygame>=2.5.0",
#  "numpy>=1.24.0",
#  "pydantic>=2.0.0"
# ]
# ///

"""
Elements module for high-tech camera overlay system.
Provides a modular framework for creating and managing UI elements.
"""

import pygame
import time
import math
import numpy as np
from pydantic import BaseModel, Field
from typing import Optional, Tuple, List, Union, Any, Dict, Callable
from enum import Enum
from abc import ABC, abstractmethod


# --| Enums and Base Types --------------
# --|----------------------------------

class Direction(Enum):
  """Direction enum for rotation and movement"""
  CW = 1  # Clockwise
  CCW = -1  # Counter-clockwise


class BlendMode(Enum):
  """Blend modes for element rendering"""
  NORMAL = 0
  ADD = pygame.BLEND_RGB_ADD
  MULTIPLY = pygame.BLEND_RGB_MULT
  SUBTRACT = pygame.BLEND_RGB_SUB
  ALPHA = pygame.BLEND_RGBA_ADD


class AnimationType(Enum):
  """Animation types for UI elements"""
  LINEAR = 0
  SINE = 1
  ELASTIC = 2
  BOUNCE = 3
  EXPONENTIAL = 4


class Color(BaseModel):
  """RGBA color representation with conversion methods"""
  r: int = Field(default=255, ge=0, le=255)
  g: int = Field(default=255, ge=0, le=255)
  b: int = Field(default=255, ge=0, le=255)
  a: int = Field(default=255, ge=0, le=255)

  def __init__(self, r, g, b, a=255, **data: Any):
    super().__init__(**data)
    self.r = int(max(0, min(255, r)))
    self.g = int(max(0, min(255, g)))
    self.b = int(max(0, min(255, b)))
    self.a = int(max(0, min(255, a)))

  def __str__(self):
    return f"Color({self.r}, {self.g}, {self.b}, {self.a})"

  def __repr__(self):
    return self.__str__()

  def to_tuple(self) -> Tuple[int, int, int, int]:
    """Convert to RGBA tuple"""
    return (self.r, self.g, self.b, self.a)

  def to_tuple_rgb(self) -> Tuple[int, int, int]:
    """Convert to RGB tuple"""
    return (self.r, self.g, self.b)

  def blend(self, other: 'Color', factor: float) -> 'Color':
    """Blend with another color using the given factor (0.0 - 1.0)"""
    factor = max(0.0, min(1.0, factor))
    inv_factor = 1.0 - factor
    return Color(
      r=int(self.r * inv_factor + other.r * factor),
      g=int(self.g * inv_factor + other.g * factor),
      b=int(self.b * inv_factor + other.b * factor),
      a=int(self.a * inv_factor + other.a * factor)
    )

  def with_alpha(self, alpha: int) -> 'Color':
    """Create a new color with the specified alpha value"""
    return Color(self.r, self.g, self.b, alpha)

  def fade(self, factor: float) -> 'Color':
    """Create a new color faded by the given factor (0.0 - 1.0)"""
    factor = max(0.0, min(1.0, factor))
    return Color(
      r=int(self.r * factor),
      g=int(self.g * factor),
      b=int(self.b * factor),
      a=int(self.a * factor)
    )

  @staticmethod
  def from_tuple(color_tuple: Tuple[int, ...]) -> 'Color':
    """Create a Color from a tuple"""
    if len(color_tuple) == 3:
      return Color(color_tuple[0], color_tuple[1], color_tuple[2])
    elif len(color_tuple) == 4:
      return Color(color_tuple[0], color_tuple[1], color_tuple[2], color_tuple[3])
    else:
      raise ValueError(f"Invalid color tuple: {color_tuple}")


# --| Base Element Class --------------
# --|----------------------------------

class Element(BaseModel, ABC):
  """Base class for all UI elements"""
  id: str = Field(default="")
  position: Tuple[float, float] = Field(default=(0, 0))
  visible: bool = Field(default=True)
  alpha: int = Field(default=255, ge=0, le=255)
  scale: float = Field(default=1.0, gt=0)
  rotation: float = Field(default=0.0)  # In radians
  blend_mode: BlendMode = Field(default=BlendMode.NORMAL)
  z_index: int = Field(default=0)

  # Surface to draw on
  _surface: Optional[pygame.Surface] = None
  _needs_redraw: bool = True

  class Config:
    arbitrary_types_allowed = True

  def __init__(self, **data):
    super().__init__(**data)
    self._frame_count = 0
    self._creation_time = time.time()

  @abstractmethod
  def update(self, dt: float) -> None:
    """Update element state"""
    pass

  @abstractmethod
  def _draw_content(self) -> None:
    """Draw the element content on its surface"""
    pass

  def draw(self, surface: pygame.Surface, position: Optional[Tuple[float, float]] = None) -> None:
    """Draw the element on the provided surface"""
    if not self.visible:
      return

    # Update frame counter
    self._frame_count += 1

    # Use provided position or element's position
    pos = position if position is not None else self.position

    # Draw content if needed
    if self._surface is None or self._needs_redraw:
      self._redraw()

    # Draw the element
    if self._surface is not None:
      # Get center as rotation pivot
      pivot = (self._surface.get_width() // 2, self._surface.get_height() // 2)

      # Create rotated and scaled copy if needed
      if self.rotation != 0 or self.scale != 1.0:
        # Create scaled and rotated copy
        scaled_size = (int(self._surface.get_width() * self.scale),
                       int(self._surface.get_height() * self.scale))

        if scaled_size[0] <= 0 or scaled_size[1] <= 0:
          return  # Skip drawing if scaled to zero size

        scaled = pygame.transform.scale(self._surface, scaled_size)

        if self.rotation != 0:
          rotated = pygame.transform.rotate(scaled, -math.degrees(self.rotation))
          draw_pos = (int(pos[0] - rotated.get_width() // 2),
                      int(pos[1] - rotated.get_height() // 2))
          surface.blit(rotated, draw_pos, special_flags=self.blend_mode.value)
        else:
          draw_pos = (int(pos[0] - scaled.get_width() // 2),
                      int(pos[1] - scaled.get_height() // 2))
          surface.blit(scaled, draw_pos, special_flags=self.blend_mode.value)
      else:
        # Just blit directly
        draw_pos = (int(pos[0] - self._surface.get_width() // 2),
                    int(pos[1] - self._surface.get_height() // 2))
        surface.blit(self._surface, draw_pos, special_flags=self.blend_mode.value)

  def _redraw(self) -> None:
    """Redraw the element on its surface"""
    # Init surface if needed
    if self._surface is None:
      size = self._get_size()
      self._surface = pygame.Surface(size, pygame.SRCALPHA)

      # Clear surface
      self._surface.fill((0, 0, 0, 0))

      # Draw content
      self._draw_content()

      # Mark as drawn
      self._needs_redraw = False

  def _get_size(self) -> Tuple[int, int]:
    """Get the element size in pixels"""
    return (100, 100)  # Default size, override in subclasses

  def invalidate(self) -> None:
    """Mark the element as needing redraw"""
    self._needs_redraw = True

  def get_age(self) -> float:
    """Get the element age in seconds"""
    return time.time() - self._creation_time


# --| Animation Helpers --------------
# --|----------------------------------

class Animation(BaseModel):
  """Handles animation of a property over time"""
  start_value: float = Field(default=0.0)
  end_value: float = Field(default=1.0)
  duration: float = Field(default=1.0)  # In seconds
  delay: float = Field(default=0.0)  # In seconds
  loop: bool = Field(default=False)
  type: AnimationType = Field(default=AnimationType.LINEAR)

  # State
  _start_time: Optional[float] = None
  _is_running: bool = False

  def start(self):
    """Start the animation"""
    self._start_time = time.time()
    self._is_running = True

  def update(self) -> float:
    """Update animation and return current value"""
    if not self._is_running:
      return self.start_value

    # Get current time
    current_time = time.time()
    elapsed = current_time - self._start_time - self.delay

    # Handle delay
    if elapsed < 0:
      return self.start_value

    # Handle completion
    if elapsed >= self.duration and not self.loop:
      self._is_running = False
      return self.end_value

    # Calculate progress (0.0 - 1.0)
    progress = (elapsed % self.duration) / self.duration if self.loop else min(1.0, elapsed / self.duration)

    # Apply easing function
    if self.type == AnimationType.SINE:
      progress = (1 - math.cos(progress * math.pi)) / 2
    elif self.type == AnimationType.ELASTIC:
      # Elastic easing
      p = 0.3
      s = p / 4
      if progress == 0 or progress == 1:
        return progress
      progress = progress - 1
      return -(2 ** (10 * progress)) * math.sin((progress - s) * (2 * math.pi) / p) + 1
    elif self.type == AnimationType.BOUNCE:
      # Bounce easing
      if progress < (1 / 2.75):
        return 7.5625 * progress * progress
      elif progress < (2 / 2.75):
        progress -= (1.5 / 2.75)
        return 7.5625 * progress * progress + 0.75
      elif progress < (2.5 / 2.75):
        progress -= (2.25 / 2.75)
        return 7.5625 * progress * progress + 0.9375
      else:
        progress -= (2.625 / 2.75)
        return 7.5625 * progress * progress + 0.984375
    elif self.type == AnimationType.EXPONENTIAL:
      # Exponential easing
      if progress == 0:
        return 0
      return 2 ** (10 * (progress - 1))

    # Linear is default
    return progress

  def get_value(self) -> float:
    """Get the current value"""
    progress = self.update()
    return self.start_value + (self.end_value - self.start_value) * progress

  def is_complete(self) -> bool:
    """Check if animation is complete"""
    return not self._is_running


# --| Rotating Elements --------------
# --|----------------------------------

class RotatingElementSegment(BaseModel):
  """A single segment of a rotating element (arc, line, etc.)"""
  id: str = Field(default="")
  start_angle: float = Field(default=0.0)  # In radians
  end_angle: float = Field(default=math.pi / 4)  # In radians
  rotation_speed: float = Field(default=0.5)  # Radians per second
  direction: Direction = Field(default=Direction.CW)
  radius: float = Field(default=50.0)  # Outer radius
  inner_radius: float = Field(default=45.0)  # Inner radius (for arcs)
  color: Color = Field(default=Color(255, 255, 255, 255))
  thickness: int = Field(default=2)
  alpha: int = Field(default=255)
  visible: bool = Field(default=True)
  dash_length: int = Field(default=0)  # 0 = solid line
  dash_gap: int = Field(default=0)  # 0 = solid line
  pulse_min: float = Field(default=1.0)  # Min pulse scale
  pulse_max: float = Field(default=1.0)  # Max pulse scale
  pulse_speed: float = Field(default=0.0)  # Pulse cycles per second
  current_angle: float = Field(default=0.0)
  offset: Tuple[float, float] = Field(default=(0.0, 0.0))
  blend_mode: BlendMode = Field(default=BlendMode.NORMAL)

  def update(self, dt: float) -> None:
    """Update segment animation"""
    if not self.visible:
      return

    # Make sure dt is reasonable (limit to max 1/10 second if too large)
    # This prevents huge jumps if the frame rate drops
    capped_dt = min(dt, 0.1)
    
    # Scale rotation_speed to be much faster 
    # (since our dt values are typically very small)
    effective_speed = self.rotation_speed * 5.0  # Scale up the speed significantly
    
    # Update rotation
    self.current_angle += effective_speed * capped_dt * self.direction.value

    # Keep angle in 0-2π range
    while self.current_angle >= math.pi * 2:
      self.current_angle -= math.pi * 2
    while self.current_angle < 0:
      self.current_angle += math.pi * 2

  def get_current_radius(self) -> float:
    """Get the current radius with pulse applied"""
    if self.pulse_speed <= 0 or self.pulse_min >= self.pulse_max:
      return self.radius

    # Calculate pulse phase (0.0 - 1.0)
    phase = (math.sin(time.time() * self.pulse_speed * math.pi * 2) + 1) / 2

    # Apply pulse to radius
    pulse_range = self.pulse_max - self.pulse_min
    return self.radius * (self.pulse_min + phase * pulse_range)

  def draw(self, surface: pygame.Surface, center: Tuple[float, float]) -> None:
    """Draw the segment with absolute time-based rotation"""
    if not self.visible or self.alpha <= 0:
      return

    # Apply offset to center
    center = (center[0] + self.offset[0], center[1] + self.offset[1])

    # Get effective color with alpha
    color = self.color.with_alpha(min(self.alpha, self.color.a))

    # Get current radius (no pulse effect)
    radius = self.radius  # Fixed radius, NO PULSING

    # Calculate time-based rotation angle
    # Use absolute system time for continuous rotation regardless of frame rate
    time_rotation = time.time() * self.rotation_speed * self.direction.value
    
    # Calculate angles - using absolute time for rotation
    start = self.start_angle + time_rotation
    end = self.end_angle + time_rotation

    # Draw segment based on inner radius
    if abs(self.inner_radius - radius) < 2:
      # This is a circular line segment
      # Convert to screen coordinates
      start_point = (
        center[0] + math.cos(start) * radius,
        center[1] + math.sin(start) * radius
      )
      end_point = (
        center[0] + math.cos(end) * radius,
        center[1] + math.sin(end) * radius
      )

      # Draw line
      if self.dash_length > 0 and self.dash_gap > 0:
        # Dashed line
        draw_dashed_line(surface, color.to_tuple_rgb(),
                         start_point, end_point,
                         self.thickness, self.dash_length, self.dash_gap)
      else:
        # Solid line
        pygame.draw.line(surface, color.to_tuple_rgb(),
                         start_point, end_point,
                         self.thickness)
    else:
      # This is an arc segment
      rect = pygame.Rect(
        center[0] - radius,
        center[1] - radius,
        radius * 2,
        radius * 2
      )
      # This is an arc segment
      rect = pygame.Rect(
        center[0] - radius,
        center[1] - radius,
        radius * 2,
        radius * 2
      )

      # Draw arc - NOTE: pygame.draw.arc uses radians in more recent versions, not degrees
      pygame.draw.arc(surface, color.to_tuple_rgb(), rect,
                      start, end, self.thickness)

      # Draw inner arc if needed
      if self.inner_radius > 0:
        inner_rect = pygame.Rect(
          center[0] - self.inner_radius,
          center[1] - self.inner_radius,
          self.inner_radius * 2,
          self.inner_radius * 2
        )
        pygame.draw.arc(surface, color.to_tuple_rgb(), inner_rect,
                        start, end, self.thickness)

        # Connect the arcs at their ends
        start_outer = (
          center[0] + math.cos(start) * radius,
          center[1] + math.sin(start) * radius
        )
        start_inner = (
          center[0] + math.cos(start) * self.inner_radius,
          center[1] + math.sin(start) * self.inner_radius
        )
        end_outer = (
          center[0] + math.cos(end) * radius,
          center[1] + math.sin(end) * radius
        )
        end_inner = (
          center[0] + math.cos(end) * self.inner_radius,
          center[1] + math.sin(end) * self.inner_radius
        )

        # Draw connecting lines
        pygame.draw.line(surface, color.to_tuple_rgb(),
                         start_outer, start_inner, self.thickness)
        pygame.draw.line(surface, color.to_tuple_rgb(),
                         end_outer, end_inner, self.thickness)


class RotatingElement(Element):
  """A composite element made of multiple rotating segments"""
  segments: List[RotatingElementSegment] = Field(default_factory=list)
  size: int = Field(default=100)  # Size in pixels

  def _get_size(self) -> Tuple[int, int]:
    """Get size for surface creation"""
    # Add margin for potential pulsing
    return (self.size * 2, self.size * 2)

  def update(self, dt: float) -> None:
    """Update all segments"""
    for segment in self.segments:
      segment.update(dt)
    self.invalidate()  # Request redraw

  def _draw_content(self) -> None:
    """Draw all segments on the surface"""
    if not self._surface:
      return

    center = (self._surface.get_width() // 2, self._surface.get_height() // 2)

    # Draw each segment
    for segment in self.segments:
      segment.draw(self._surface, center)

  def add_segment(self, id: str = "", **kwargs) -> RotatingElementSegment:
    """Add a new segment with given properties"""
    segment = RotatingElementSegment(id=id, **kwargs)
    self.segments.append(segment)
    self.invalidate()
    return segment

  def get_segment(self, id: str) -> Optional[RotatingElementSegment]:
    """Find segment by ID"""
    for segment in self.segments:
      if segment.id == id:
        return segment
    return None

  def remove_segment(self, id: str) -> bool:
    """Remove segment by ID"""
    for i, segment in enumerate(self.segments):
      if segment.id == id:
        self.segments.pop(i)
        self.invalidate()
        return True
    return False


# --| Helper Functions ---------------
# --|----------------------------------

def draw_dashed_line(surface, color, start_pos, end_pos, width=1, dash_length=10, gap_length=10):
  """Draw a dashed line on the surface"""
  x1, y1 = start_pos
  x2, y2 = end_pos

  # Calculate line length and direction vector
  dx = x2 - x1
  dy = y2 - y1
  length = math.sqrt(dx * dx + dy * dy)

  if length < 1:  # Too short to draw
    return

  # Normalize direction vector
  dx, dy = dx / length, dy / length

  # Calculate number of segments
  dash_gap = dash_length + gap_length
  n_segments = int(length / dash_gap)

  # Draw each dash
  for i in range(n_segments + 1):
    start = i * dash_gap
    end = min(start + dash_length, length)

    # Calculate dash endpoints
    dash_start = (x1 + dx * start, y1 + dy * start)
    dash_end = (x1 + dx * end, y1 + dy * end)

    # Draw the dash
    pygame.draw.line(surface, color, dash_start, dash_end, width)


# --| Tracking Element ---------------
# --|----------------------------------

class TrackingElement(Element):
  """Base class for elements that track a target position"""

  target_position: Optional[Tuple[float, float]] = None
  current_position: Optional[Tuple[float, float]] = None

  # Smoothing parameters for tracking
  smoothing_factor: float = Field(default=0.15)
  min_speed: float = Field(default=0.5)
  max_speed: float = Field(default=30.0)
  dampening_factor: float = Field(default=0.8)
  lerp_factor: float = Field(default=0.3)

  # Advanced tracking state
  velocity: Tuple[float, float] = Field(default=(0.0, 0.0))
  last_positions: List[Tuple[float, float]] = Field(default_factory=list)
  max_position_history: int = Field(default=5)
  active: bool = Field(default=False)
  lerp_target: Optional[Tuple[float, float]] = None

  def update(self, dt: float) -> None:
    """Update tracking and animation"""
    self._update_tracking(dt)
    self._update_animation(dt)

    # Set element position to tracked position
    if self.current_position:
      self.position = self.current_position

    # Mark for redraw
    self.invalidate()

  def _update_tracking(self, dt: float) -> None:
    """Update tracking position using physics-based smoothing"""
    if not self.active or not self.target_position:
      return

    # Initialize current position if not set
    if self.current_position is None:
      self.current_position = self.target_position
      self.velocity = (0.0, 0.0)
      return

    # Update using physical model
    tx, ty = self.lerp_target or self.target_position
    cx, cy = self.current_position
    vx, vy = self.velocity

    # Calculate distance to target
    dx = tx - cx
    dy = ty - cy
    distance = math.sqrt(dx * dx + dy * dy)

    # Apply spring-based physics for smoothing
    # Adjust force based on distance (stronger pull when far away)
    force_factor = self.smoothing_factor * (1.0 + min(1.0, distance / 100.0))

    # Calculate spring force
    fx = dx * force_factor
    fy = dy * force_factor

    # Update velocity with spring force (include damping)
    vx = vx * 0.8 + fx
    vy = vy * 0.8 + fy

    # Limit speed
    speed = math.sqrt(vx * vx + vy * vy)
    if speed > 0:
      if speed < self.min_speed and distance > 1.0:
        scale_factor = self.min_speed / speed
        vx *= scale_factor
        vy *= scale_factor
      elif speed > self.max_speed:
        scale_factor = self.max_speed / speed
        vx *= scale_factor
        vy *= scale_factor

      # Update position
      nx = cx + vx * dt * 60.0  # Scale by time and normalize to 60fps
      ny = cy + vy * dt * 60.0

      # Store updated values
      self.current_position = (nx, ny)
      self.velocity = (vx, vy)

  def _update_animation(self, dt: float) -> None:
    """Override in subclasses to add custom animation"""
    pass

  def track(self, target_position: Optional[Tuple[float, float]]) -> None:
    """Start tracking a new target position"""
    # If no target, deactivate
    if target_position is None:
      self.active = False
      self.lerp_target = None
      return

    # Apply position dampening to reduce jitter
    if self.last_positions and self.dampening_factor > 0:
      # Calculate average of recent positions
      if len(self.last_positions) >= self.max_position_history:
        avg_x = sum(p[0] for p in self.last_positions) / len(self.last_positions)
        avg_y = sum(p[1] for p in self.last_positions) / len(self.last_positions)

        # Apply dampening
        dampened_target = (
          target_position[0] * (1 - self.dampening_factor) + avg_x * self.dampening_factor,
          target_position[1] * (1 - self.dampening_factor) + avg_y * self.dampening_factor
        )

        # Calculate displacement
        dx = abs(target_position[0] - avg_x)
        dy = abs(target_position[1] - avg_y)

        # Apply adaptive dampening
        if dx < 5 and dy < 5:
          self.target_position = dampened_target
        else:
          # Less dampening for larger movements
          blend_factor = min(1.0, max(0.2, (dx + dy) / 20.0))
          self.target_position = (
            dampened_target[0] * (1 - blend_factor) + target_position[0] * blend_factor,
            dampened_target[1] * (1 - blend_factor) + target_position[1] * blend_factor
          )
      else:
        self.target_position = target_position
    else:
      self.target_position = target_position

    # Apply lerp for smooth transitions
    if self.lerp_target is None:
      self.lerp_target = self.target_position
    else:
      # Calculate adaptive lerp
      lt_x, lt_y = self.lerp_target
      tp_x, tp_y = self.target_position
      distance = math.sqrt((tp_x - lt_x) ** 2 + (tp_y - lt_y) ** 2)

      # Faster for larger movements
      adaptive_factor = min(1.0, self.lerp_factor * (1.0 + min(3.0, distance / 100.0)))

      # Lerp to smooth transition
      self.lerp_target = (
        lt_x + (tp_x - lt_x) * adaptive_factor,
        lt_y + (tp_y - lt_y) * adaptive_factor
      )

    # Update histo  ry
    self.last_positions.append(self.target_position)
    if len(self.last_positions) > self.max_position_history:
      self.last_positions.pop(0)

    self.active = True

  def lerp(self, start, end, t):
    """Linear interpolation between two points"""
    return (
      start[0] + (end[0] - start[0]) * t,
      start[1] + (end[1] - start[1]) * t
    )


# --| Reticle Element ----------------
# --|----------------------------------

class ReticleElement(TrackingElement):
  """A tracking reticle composed of multiple rotating elements"""
  size: int = Field(default=50)
  color: Color = Field(default=Color(255, 50, 50, 255))
  line_thickness: int = Field(default=2)
  animation_speed: float = Field(default=0.1)

  # Reticle parts
  rotating_elements: List[RotatingElement] = Field(default_factory=list)
  pulse_speed: float = Field(default=2.0)
  pulse_min: float = Field(default=0.9)
  pulse_max: float = Field(default=1.1)
  pulse_value: float = Field(default=0.0)

  def __init__(self, **data):
    super().__init__(**data)
    self._setup_default_elements()

  def _setup_default_elements(self):
    """Setup simple reticle elements with DIRECT TIME-BASED ROTATION"""
    # Colors
    main_color = self.color  # Main color (red)
    blue_color = Color(r=50, g=150, b=255, a=self.color.a)  # Secondary color (blue)
    
    # Create a completely new implementation using time-based rotation
    # Static elements (no rotation)
    static_elements = RotatingElement(
      id="static_elements",
      size=int(self.size),
      position=(0, 0),
    )
    
    # Center dot - static
    static_elements.add_segment(
      id="center_dot",
      start_angle=0,
      end_angle=math.pi * 2,  # Full circle
      rotation_speed=0,  # No rotation
      radius=self.size * 0.08,
      inner_radius=0,  # Filled circle
      color=main_color,
      thickness=self.line_thickness,
      pulse_min=1.0,  # NO PULSING
      pulse_max=1.0,
    )
    
    # Create a special fast-rotating outer ring
    outer_ring = RotatingElement(
      id="outer_ring",
      size=int(self.size),
      position=(0, 0),
    )
    
    # Segments will use time-based rotation directly in draw call
    for i in range(4):
      angle = i * math.pi / 2  # 4 segments evenly spaced
      outer_ring.add_segment(
        id=f"outer_segment_{i}",
        start_angle=angle - 0.7,  # Wide segments
        end_angle=angle + 0.7,
        rotation_speed=5.0,  # Extra fast - time-based effect
        direction=Direction.CW,  # Clockwise
        radius=self.size * 0.9,  # Near edge
        inner_radius=self.size * 0.85,
        color=main_color,
        thickness=self.line_thickness,
        pulse_min=1.0,  # NO PULSING
        pulse_max=1.0,
      )
    
    # Create a special fast-rotating inner ring
    inner_ring = RotatingElement(
      id="inner_ring",
      size=int(self.size * 0.7),
      position=(0, 0),
    )
    
    # Inner ring segments
    for i in range(4):
      angle = i * math.pi / 2  # 4 segments evenly spaced
      inner_ring.add_segment(
        id=f"inner_segment_{i}",
        start_angle=angle - 0.7,
        end_angle=angle + 0.7,
        rotation_speed=7.0,  # Even faster than outer ring
        direction=Direction.CCW,  # Counter-clockwise
        radius=self.size * 0.7,
        inner_radius=self.size * 0.65,
        color=blue_color,
        thickness=self.line_thickness,
        pulse_min=1.0,  # NO PULSING
        pulse_max=1.0,
      )
    
    # Add all elements to the reticle
    self.rotating_elements.append(static_elements)  # Static elements
    self.rotating_elements.append(outer_ring)       # Outer rotating ring
    self.rotating_elements.append(inner_ring)       # Inner rotating ring

  def _update_animation(self, dt: float) -> None:
    """Update all reticle animations - NO PULSING, only rotation"""
    # Keep scale constant - NO PULSING
    self.scale = 1.0
    
    # Update all rotating elements
    for element in self.rotating_elements:
      element.update(dt)

  def _get_size(self) -> Tuple[int, int]:
    """Get size for surface creation"""
    # Add margin for pulse
    margin = self.size * self.pulse_max * 1.2
    return (int(margin * 2), int(margin * 2))

  def _draw_content(self) -> None:
    """Draw the reticle elements"""
    if not self._surface:
      return

    center = (self._surface.get_width() // 2, self._surface.get_height() // 2)

    # Draw all elements
    for element in self.rotating_elements:
      element.draw(self._surface, center)


# --| Factory Functions ---------------
# --|----------------------------------

def create_rotating_reticle(size=50, color=(255, 50, 50), thickness=2, **kwargs) -> ReticleElement:
  """Create a reticle with rotating elements"""
  # Convert tuple or list color to Color object if needed
  if not isinstance(color, Color):
    if isinstance(color, (tuple, list)) and len(color) >= 3:
      # Create a Color object with the RGB values
      if len(color) >= 4:
        color = Color(r=color[0], g=color[1], b=color[2], a=color[3])
      else:
        color = Color(r=color[0], g=color[1], b=color[2], a=255)
    else:
      # Default color if invalid format
      color = Color(r=255, g=50, b=50, a=255)

  # Create reticle
  reticle = ReticleElement(
    size=size,
    color=color,
    line_thickness=thickness,
    **kwargs
  )

  return reticle
