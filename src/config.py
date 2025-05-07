#!/usr/bin/env python

"""
Configuration module for the camera overlay system.
Handles loading configuration from YAML files or creating defaults.
"""

import os
import sys
import yaml
from pydantic import BaseModel, Field
from typing import Optional, Tuple, List, Union
from pathlib import Path


class OverlayConfig(BaseModel):
  """Configuration model for the camera overlay system."""

  # Application settings
  title: str = "High-Tech Camera Overlay"
  source: Union[str, int] = 0
  max_fps: int = 30
  display_width: Optional[int] = None
  display_height: Optional[int] = None

  # Camera settings
  camera_width: Optional[int] = None
  camera_height: Optional[int] = None

  max_input_width: Optional[int] = 1280
  max_input_height: Optional[int] = 720
  maintain_aspect_ratio: bool = True

  # Video orientation settings
  flip_horizontal: bool = False
  flip_vertical: bool = False
  rotation_degrees: int = 0  # 0, 90, 180, or 270

  track_every_n_frames: int = 10

  # Detection settings
  face_detection_model: str = "face_detection_model.tflite"
  face_detection_threshold: float = 0.5

  object_detection_model: str = "object_detection_model.tflite"
  object_detection_threshold: float = 0.5

  hand_detection_model: str = "hand_detection_model.tflite"
  hand_detection_threshold: float = 0.5

  pose_detection_model: str = "pose_detection_model.tflite"
  pose_detection_threshold: float = 0.5

  # Grid settings
  grid_cols: int = 9
  grid_rows: int = 5
  grid_color: Tuple[int, int, int, int] = (0, 255, 255, 150)  # Cyan with alpha
  grid_line_thickness: int = 1
  grid_animation_speed: float = 0.2

  # Grid cell settings
  cell_highlight_alpha: int = 50
  cell_active_color: Tuple[int, int, int] = (255, 200, 50)
  cell_inactive_color: Tuple[int, int, int] = (50, 200, 255)

  # Motion detection settings
  motion_sensitivity: float = 0.3
  motion_blur: int = 5
  motion_threshold: int = 25

  # Reticle settings
  reticle_size: int = 50
  reticle_color: List[int] = Field(default=[255, 50, 50])  # (255, 50, 50) # Red
  reticle_line_thickness: int = 2
  reticle_animation_speed: float = 0.1

  # Reticle rotating elements settings
  reticle_element_speeds: List[float] = Field(default=[0.08, 0.05, 0.06, 0.03])
  reticle_element_directions: List[int] = Field(default=[1, -1, 1, -1])  # 1=clockwise, -1=counter-clockwise
  reticle_element_colors: List[List[int]] = Field(default=[[255, 50, 50], [50, 255, 50], [50, 50, 255], [255, 255, 50]])

  use_simple_tracker: bool = True

  # Face tracking persistence settings
  max_face_detections: int = 10
  face_persistence_frames: int = 120  # Number of frames to keep tracking a face after it disappears
  face_history_size: int = 30  # Size of face tracking history buffer

  # Reticle movement smoothing
  reticle_smoothing_factor: float = 0.15  # Lower = smoother but slower tracking (0.05-0.3)
  reticle_smoothing_min_speed: float = 0.5  # Minimum movement speed in pixels per frame
  reticle_smoothing_max_speed: float = 30.0  # Maximum movement speed in pixels per frame
  reticle_dampening_factor: float = 0.8  # Micro-movement dampening (0.0-1.0)
  reticle_lerp_factor: float = 0.2  # Linear interpolation factor (lower = smoother transitions)

  # Info panel settings
  info_panel_width: int = 200
  info_panel_height: int = 120
  info_panel_header_color: Tuple[int, int, int, int] = (25, 25, 25, 100)
  info_panel_color: Tuple[int, int, int, int] = (10, 10, 50, 200)  # Dark blue with alpha
  info_panel_text_color: Tuple[int, int, int] = (0, 255, 255, 200)  # Cyan
  info_panel_font_size: int = 14
  info_panel_offset_x: int = 60
  info_panel_offset_y: int = -60

  # Animation settings
  animation_speed: float = 0.1

def load_config(config_path: str = None) -> OverlayConfig:
  """
  Load configuration from YAML file or create default.

  Args:
      config_path: Path to configuration YAML file

  Returns:
      OverlayConfig: Configuration object
  """
  # Create default config
  config = OverlayConfig()

  # If config path is provided, try to load from file
  if config_path and os.path.exists(config_path):
    try:
      with open(config_path, 'r') as f:
        yaml_config = yaml.safe_load(f)

      # Update config with values from YAML
      if yaml_config:
        for key, value in yaml_config.items():
          if hasattr(config, key):
            setattr(config, key, value)

    except Exception as e:
      print(f"Error loading configuration from {config_path}: {e}")
      print("Using default configuration")

  return config


def save_config(config: OverlayConfig, config_path: str = "config.yaml"):
  """
  Save configuration to YAML file.

  Args:
      config: Configuration object
      config_path: Path to save configuration YAML file
  """
  try:
    # Convert config to dict
    config_dict = config.dict()

    # Save to YAML
    with open(config_path, 'w') as f:
      yaml.dump(config_dict, f, default_flow_style=False)

    print(f"Configuration saved to {config_path}")

  except Exception as e:
    print(f"Error saving configuration to {config_path}: {e}")


def create_default_config(config_path: str = "config.yaml"):
  """
  Create and save default configuration to YAML file.

  Args:
      config_path: Path to save configuration YAML file
  """
  # Create default config
  config = OverlayConfig()

  # Save to YAML
  save_config(config, config_path)

  return config
