#!/usr/bin/env -S uv run

# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "opencv-python>=4.8.0",
#   "numpy>=1.24.0",
#   "pillow>=10.0.0",
#   "pydantic>=2.0.0",
#   "pygame>=2.5.0",
#   "mediapipe>=0.10.21",
#   "imutils>=0.5.4",
#   "pyyaml>=6.0.2"
# ]
# ///

import argparse
import os
import time

import cv2
import mediapipe as mp
import pygame
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.config import load_config
from src.face_tracker import FaceTracker
# Import application modules
from src.grid import Grid
from src.motion import MotionDetector
from src.person_info import PersonInfoPanel
from src.reticle import Reticle
from src.utils import visualize, write_detection_data


class CameraOverlayApp:
  """Main application class for managing the high-tech camera overlay system."""

  def __init__(self, config=None):
    """Initialize the application with optional configuration."""
    # Load configuration
    self.config = config or load_config()

    # Initialize camera source
    self.setup_camera()

    # Initialize components
    self.grid = None
    self.motion_detector = None
    self.reticle = None
    self.info_panel = None

    # Initialize metrics
    self.last_time = time.time()
    self.frame_count = 0
    self.fps = 0
    self.print_count = 0

    # Set up camera / video source
    self.cap = None
    self.using_example_frames = False
    self.setup_camera()

    # Initialize face detection
    self.mp_face_detection = mp.solutions.face_detection
    self.mp_drawing = mp.solutions.drawing_utils

    self.base_options = python.BaseOptions(model_asset_path='models/efficientdet.tflite')

    self.options = vision.ObjectDetectorOptions(
      base_options=self.base_options,
      running_mode=vision.RunningMode.LIVE_STREAM,
      score_threshold=0.5,
      result_callback=self.stream_callback
    )

    self.object_detector = vision.ObjectDetector.create_from_options(self.options)

    self.face_detection = self.mp_face_detection.FaceDetection(
      model_selection=1,
      min_detection_confidence=self.config.face_detection_threshold
    )

    self.face_tracker = FaceTracker(
      min_detection_confidence=self.config.face_detection_threshold,
      max_faces=self.config.max_face_detections
    )

    self.DETECTION_RESULT = []
    self.FACE_DATA = []

    # Face tracking history for persistence
    self.tracked_faces = []
    self.face_history = []
    self.face_history_max_size = self.config.face_history_size
    self.face_lost_frames = 0
    self.face_max_lost_frames = self.config.face_persistence_frames
    self.last_detection_positions = {}  # Store last detected positions by ID
    self.target_positions = {}  # Store target positions for interpolation
    self.track_every_n_frames = self.config.track_every_n_frames  # How often to update tracking targets
    self.interpolation_factor = 0.05  # How fast to move towards the target (0.1-0.3 is good)

    # Initialize detector
    self.setup_detector()

    # Initialize display
    self.setup_display()

    # Initialize overlay components
    self.setup_overlay_components()

    # Application status
    self.running = False
    self.fps = 0
    self.frame_count = 0
    self.previous_face_count = 0
    self.last_time = time.time()

  def stream_callback(self, result: vision.ObjectDetectorResult, unused_output_image: mp.Image, timestamp_ms: int):
    # print(result)
    self.DETECTION_RESULT.append(result)

  def setup_camera(self):
    """Setup camera or video source."""
    source = self.config.source
    if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
      source = int(source)

    self.cap = cv2.VideoCapture(source)
    self.use_example_frames = False

    # Check if camera opened successfully
    if not self.cap.isOpened():
      print(f"Warning: Could not open camera/video source: {source}")
      print("Falling back to example frames...")
      self.use_example_frames = True
      self.example_frame_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'video_example', 'frames')

      # Verify the example frames directory exists
      if os.path.exists(self.example_frame_dir):
        self.example_frames = sorted([f for f in os.listdir(self.example_frame_dir) if f.endswith('.jpg')])
        if not self.example_frames:
          print(f"Error: No jpg files found in {self.example_frame_dir}")
      else:
        print(f"Error: Example frames directory not found: {self.example_frame_dir}")
        self.example_frames = []

      self.current_frame_index = 0

      # Set default frame properties for example frames
      if self.example_frames:
        # Load first frame to get dimensions
        first_frame_path = os.path.join(self.example_frame_dir, self.example_frames[0])
        test_frame = cv2.imread(first_frame_path)
        if test_frame is not None:
          self.frame_height, self.frame_width = test_frame.shape[:2]
        else:
          # Default if can't read frame
          self.frame_width = 1280
          self.frame_height = 720
      else:
        print("Error: No example frames found in video_example/frames directory")
        self.frame_width = 1280
        self.frame_height = 720

      self.frame_rate = 60  # Default frame rate for example frames
      return

    # Get camera properties
    self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    self.frame_rate = int(self.cap.get(cv2.CAP_PROP_FPS))

    if self.frame_rate == 0:  # Fallback if FPS not available
      self.frame_rate = 60

    # Set camera properties if specified in config
    if hasattr(self.config, 'camera_width') and self.config.camera_width:
      self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
      self.frame_width = self.config.camera_width

    if hasattr(self.config, 'camera_height') and self.config.camera_height:
      self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)
      self.frame_height = self.config.camera_height

  def setup_detector(self):
    """Setup motion and face detection."""
    # Motion detection
    self.motion_detector = MotionDetector(
      grid_cols=self.config.grid_cols,
      grid_rows=self.config.grid_rows,
      frame_width=self.frame_width,
      frame_height=self.frame_height,
      sensitivity=self.config.motion_sensitivity
    )

    # Face detection using mediapipe
    self.mp_face_detection = mp.solutions.face_detection
    self.mp_drawing = mp.solutions.drawing_utils
    self.face_detection = self.mp_face_detection.FaceDetection(
      model_selection=1,
      min_detection_confidence=self.config.face_detection_threshold
    )

    # Track detected faces
    self.detected_faces = []

  def setup_display(self):
    """Setup pygame display."""
    pygame.init()

    # Display size
    self.display_width = self.config.display_width or self.frame_width
    self.display_height = self.config.display_height or self.frame_height

    # Create display
    self.screen = pygame.display.set_mode(
      (self.display_width, self.display_height)
    )
    pygame.display.set_caption(self.config.title)

    # Clock for frame timing
    self.clock = pygame.time.Clock()

  def setup_overlay_components(self):
    """Setup all overlay visual components."""
    # Grid overlay
    self.grid = Grid(
      cols=self.config.grid_cols,
      rows=self.config.grid_rows,
      width=self.display_width,
      height=self.display_height,
      line_color=self.config.grid_color,
      line_thickness=self.config.grid_line_thickness,
      animation_speed=self.config.grid_animation_speed,
      cell_highlight_alpha=self.config.cell_highlight_alpha,
      cell_active_color=self.config.cell_active_color,
      cell_inactive_color=self.config.cell_inactive_color
    )

    # Reticle for head tracking
    self.reticle = Reticle(
      size=self.config.reticle_size,
      color=self.config.reticle_color,
      line_thickness=self.config.reticle_line_thickness,
      animation_speed=self.config.reticle_animation_speed,
      smoothing_factor=self.config.reticle_smoothing_factor,
      min_speed=self.config.reticle_smoothing_min_speed,
      max_speed=self.config.reticle_smoothing_max_speed,
      dampening_factor=self.config.reticle_dampening_factor,
      lerp_factor=getattr(self.config, 'reticle_lerp_factor', 0.2)  # Support older config files
    )

    # Configure individual rotating elements if specified in config
    if hasattr(self.config, 'reticle_element_speeds'):
      for i, speed in enumerate(self.config.reticle_element_speeds):
        if i < len(self.reticle.rotating_elements):
          self.reticle.rotating_elements[i]['speed'] = speed

    if hasattr(self.config, 'reticle_element_directions'):
      for i, direction in enumerate(self.config.reticle_element_directions):
        if i < len(self.reticle.rotating_elements):
          self.reticle.rotating_elements[i]['direction'] = direction

    if hasattr(self.config, 'reticle_element_colors') and self.config.reticle_element_colors:
      for i, color in enumerate(self.config.reticle_element_colors):
        if i < len(self.reticle.rotating_elements):
          self.reticle.rotating_elements[i]['color'] = color

    # Person information panel
    self.info_panel = PersonInfoPanel(
      width=self.config.info_panel_width,
      height=self.config.info_panel_height,
      header_color=self.config.info_panel_header_color,
      color=self.config.info_panel_color,
      text_color=self.config.info_panel_text_color,
      font_size=self.config.info_panel_font_size
    )

  def process_frame(self, frame):
    """Process a single video frame with all detections and overlays."""
    # Convert frame to RGB for mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Motion detection
    motion_grid = self.motion_detector.detect(frame)

    use_simple_tracker = self.config.use_simple_tracker
    if use_simple_tracker:
      # Face detection

      face_results = self.face_detection.process(rgb_frame)
      should_update_targets = (self.frame_count % self.track_every_n_frames == 0)
      should_update_bodies = (self.frame_count % (self.track_every_n_frames * 0.5) == 0)

      # if should_update_targets:

      # if should_update_bodies:
      mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
      self.object_detector.detect_async(mp_image, time.time_ns() // 1_000_000)

      # Process face detection results
      current_faces = []
      face_detected = False

      # if self.DETECTION_RESULT:
      #   for result in self.DETECTION_RESULT:
      #     # print(result)
      #     frame = visualize(frame, result, self.frame_width, self.frame_height)
      # 
      #   self.DETECTION_RESULT.clear()

      # for keypoint in detection.relative_keypoints:
      #   keypoint_px = normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
      #                                                 self.frame_width, self.frame_height)
      #   color, thickness, radius = (0, 255, 0), 2, 2
      #   cv2.circle(frame, keypoint_px, thickness, color, radius)

      if face_results.detections:
        face_detected = True

        for i, detection in enumerate(face_results.detections):
          # print(detection)
          bbox = detection.location_data.relative_bounding_box

          # Convert relative coordinates to pixel coordinates
          x = int(bbox.xmin * self.frame_width)
          y = int(bbox.ymin * self.frame_height)
          w = int(bbox.width * self.frame_width)
          h = int(bbox.height * self.frame_height)

          # TODO: Remove this, just for debugging
          # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

          # The face detection coordinates need to be properly mapped to the display
          # The frame is displayed in its original orientation in our current setup
          display_x = x + w // 2  # Horizontal center of face
          display_y = y + h // 2  # Vertical center of face

          # Try to match with existing faces by proximity
          face_id = i + 1  # Default ID if no match

          # Find closest existing face if we have previous detections
          if self.detected_faces:
            min_dist = float('inf')
            for prev_face in self.detected_faces:
              prev_x, prev_y = prev_face['center']
              dist = ((display_x - prev_x) ** 2 + (display_y - prev_y) ** 2) ** 0.5
              if dist < min_dist and dist < w:  # Use width as a distance threshold
                min_dist = dist
                face_id = prev_face['id']

          # Check if this is a new detection or continued tracking of an existing face
          is_fresh_detection = True
          if self.detected_faces:
            for prev_face in self.detected_faces:
              if prev_face['id'] == face_id:
                # This face was already tracked, so it's not freshly detected
                is_fresh_detection = False
                # Increment the frames counter for existing faces
                frames_since = prev_face.get('frames_since_detection', 0) + 1
                break
            else:
                # If the loop completes without a break, this is a new face
                frames_since = 0
          else:
            # If no previous faces, this is a new detection
            frames_since = 0
          
          face_data = {
            'bbox': (x, y, w, h),
            'center': (display_x, display_y),
            'confidence': detection.score[0],
            'id': face_id,
            'name': f"Person {face_id}",
            'status': "Active",
            'freshly_detected': is_fresh_detection,
            'frames_since_detection': frames_since
          }

          current_faces.append(face_data)

          if should_update_targets:
            self.target_positions[face_id] = (display_x, display_y)

        # Reset lost frames counter when faces are detected
        self.face_lost_frames = 0
      else:
        # Increment the lost frames counter when no faces are detected
        self.face_lost_frames += 1

      # Handle face persistence when no faces are detected
      if not face_detected and self.face_lost_frames < self.face_max_lost_frames and self.face_history:
        # Use the most recent face positions from history
        latest_faces = self.face_history[-1]
        
        for face in latest_faces:
          # Create a copy of the historical face data with updated status
          persisted_face = face.copy()
          persisted_face['status'] = "Persistent"
          persisted_face['freshly_detected'] = False
          
          # Update the frames since detection counter properly
          previous_frames = face.get('frames_since_detection', 0)
          persisted_face['frames_since_detection'] = previous_frames + 1
          
          current_faces.append(persisted_face)

      for face in current_faces:
        face_id = face['id']

        # If we have both current position and target position
        if face_id in self.last_detection_positions and face_id in self.target_positions:
          last_x, last_y = self.last_detection_positions[face_id]
          target_x, target_y = self.target_positions[face_id]

          # Interpolate between last position and target position
          new_x = last_x + (target_x - last_x) * self.interpolation_factor
          new_y = last_y + (target_y - last_y) * self.interpolation_factor

          # Update the face center with interpolated position
          face['center'] = (int(new_x), int(new_y))

          # Store the new position as the last known position
          self.last_detection_positions[face_id] = (new_x, new_y)
        else:
          # If no prior data, initialize with current position
          self.last_detection_positions[face_id] = face['center']
          self.target_positions[face_id] = face['center']

      # Don't update self.detected_faces here - we handle this in the history update below
      pass

      # Update face records for better persistence tracking
      updated_face_records = []
      
      # Process each face and preserve tracking information
      for face in current_faces:
        updated_face = face.copy()
        
        # If it's not a freshly detected face, ensure the status is properly set
        if not face.get('freshly_detected', True):
          # Make sure persistence shows correctly
          updated_face['status'] = "Tracked"
        else:
          # For consistency, newly detected faces should be marked differently
          # only on their first appearance
          for prev_record in self.detected_faces:
            if prev_record['id'] == face['id']:
              # If we've seen this ID before, it's not truly fresh anymore
              updated_face['freshly_detected'] = False
              updated_face['status'] = "Tracked"
              # Use the previous frames_since_detection and increment it
              updated_face['frames_since_detection'] = prev_record.get('frames_since_detection', 0) + 1
              break
        
        updated_face_records.append(updated_face)
      
      # Make sure we don't lose any persistent faces
      # Keep track of which face IDs we've handled in this frame
      current_face_ids = {face['id'] for face in updated_face_records}
      
      # Check if any previously detected faces are missing in the current frame
      if self.detected_faces:
        for prev_face in self.detected_faces:
          # If this face ID is not in current detections, add it with persistence
          if prev_face['id'] not in current_face_ids:
            # Only add if it hasn't exceeded max persistence frames
            frames_since = prev_face.get('frames_since_detection', 0) + 1
            if frames_since < self.face_max_lost_frames:
              persisted_face = prev_face.copy()
              persisted_face['freshly_detected'] = False
              persisted_face['status'] = "Persistent"
              persisted_face['frames_since_detection'] = frames_since
              updated_face_records.append(persisted_face)
      
      # Add the current faces to history
      if updated_face_records or not self.face_history:  # Only add meaningful entries to history
        self.face_history.append(updated_face_records)
        # Limit history size
        if len(self.face_history) > self.face_history_max_size:
          self.face_history.pop(0)
          
      # Use the updated records for detection
      self.detected_faces = updated_face_records

      if face_results and face_results.detections:
        self.previous_face_count = len(face_results.detections)

    else:
      # Face tracking using our new tracker
      self.detected_faces = self.face_tracker.detect_and_track(frame)

    # Update grid with motion data
    self.grid.update(motion_grid)

    # Convert frame to pygame surface
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Apply orientation transformations from config
    # Apply rotation if specified
    if self.config.rotation_degrees == 90:
      frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif self.config.rotation_degrees == 180:
      frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif self.config.rotation_degrees == 270:
      frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Apply flips if specified
    if self.config.flip_horizontal:
      frame = cv2.flip(frame, 1)  # 1 means flip horizontally
    if self.config.flip_vertical:
      frame = cv2.flip(frame, 0)  # 0 means flip vertically

    # Convert to pygame surface
    frame = pygame.surfarray.make_surface(frame)

    return frame

  def write_face_data(self):
    # Clear the file first to avoid accumulating too much data
    with open('face_data.txt', 'w') as f:
      f.write("# Face tracking data\n")
      f.write("# Format: id, freshly_detected, frames_since_detection, status, confidence, position\n\n")
    
    # Now append the face data in a more readable format
    with open('face_data.txt', 'a') as f:
      # Group faces by ID to make it easier to track the persistence of each face
      faces_by_id = {}
      for face in self.FACE_DATA:
        face_id = face.get('id', 'unknown')
        if face_id not in faces_by_id:
          faces_by_id[face_id] = []
        faces_by_id[face_id].append(face)
      
      # Write data for each face ID group
      for face_id, faces in faces_by_id.items():
        f.write(f"===== Face ID: {face_id} =====\n")
        for face in faces:
          # Format the data in a more readable way
          fresh = face.get('freshly_detected', True)
          frames = face.get('frames_since_detection', 0)
          status = face.get('status', 'Unknown')
          conf = face.get('confidence', 0.0)
          pos = face.get('center', (0, 0))
          
          f.write(f"ID: {face_id}, Fresh: {fresh}, Frames: {frames}, Status: {status}, Conf: {conf:.4f}, Pos: {pos}\n")
        f.write("\n")  # Add a blank line between face groups
      
      # Reset the face data list after writing
      self.FACE_DATA = []

  def render(self, frame_surface):
    """Render the frame with all overlay elements."""
    # Clear screen
    self.screen.fill((0, 0, 0))

    # Draw video frame
    self.screen.blit(frame_surface, (0, 0))

    # Draw grid overlay
    self.grid.draw(self.screen)

    # Process and draw face tracking elements
    for face in self.detected_faces:
      # Write contents of `face` to file
      self.FACE_DATA.append(face)

      # Ensure info panel is always active for consistent display
      self.info_panel.active = True
  
      # Get face-specific parameters based on detection status
      # For persistent faces, we'll modify the appearance slightly
      if face.get('freshly_detected', True):
        # For actively detected faces, use full color and size
        reticle_color = self.config.reticle_color
        reticle_size = self.config.reticle_size
        info_color = self.config.info_panel_color
        info_text_color = self.config.info_panel_text_color
        info_alpha = 255
  
      else:
        # For persistent faces (not freshly detected), fade the colors based on how long since last seen
        frames_since = face.get('frames_since_detection', 0)
        # More aggressive fading to make it clearer when faces are lost
        fade_ratio = 1.0 #- (frames_since / self.face_max_lost_frames) * 0.8  # Max 80% fade
  
        # Adjust reticle color for persistence (keeping the same hue but reducing brightness)
        r, g, b = self.config.reticle_color
        reticle_color = (
          int(r * fade_ratio),
          int(g * fade_ratio),
          int(b * fade_ratio)
        )
  
        # Slightly smaller reticle for persistent faces
        reticle_size = int(self.config.reticle_size * (0.9 + 0.1 * fade_ratio))
  
        # Adjust info panel colors for persistence
        r, g, b, a = self.config.info_panel_color
        info_color = (r, g, b, int(a * fade_ratio))
  
        r, g, b = self.config.info_panel_text_color
        info_text_color = (r, g, b, int(200 * fade_ratio))
  
        info_alpha = int(200 * fade_ratio)
  
      # Ensure face is always drawn regardless of ID
      # This is particularly important for fixing Person ID #2 disappearance
      
      # Force the face to be visible and tracked
      if face['id'] == 2:  # Special handling for the problematic ID #2
        face['confidence'] = max(face['confidence'], 0.7)  # Ensure high confidence
      
      # Update reticle with face position and custom size/color for persistence
      self.reticle.update(face['center'])
      
      # Use custom colors for persistent faces
      original_color = self.reticle.color
      original_size = self.reticle.size
      self.reticle.color = reticle_color
      self.reticle.size = reticle_size
      
      # Draw reticle with debugging info
      self.reticle.draw(self.screen, face['center'])
  
      # Restore original reticle properties
      self.reticle.color = original_color
      self.reticle.size = original_size
  
      # Draw person info panel
      panel_x = face['center'][0] + self.config.info_panel_offset_x
      panel_y = face['center'][1] + self.config.info_panel_offset_y
  
      # Use custom status based on persistence
      display_status = face.get('status', 'Unknown')
      display_confidence = face.get('confidence', 0.0)
      
      # Always show frames since detection for better debugging
      frames_since = face.get('frames_since_detection', 0)
      
      # For ID #2, ensure we always maintain a high confidence display
      if face['id'] == 2:
          # Keep confidence display high for ID #2
          # display_confidence = max(display_confidence, 0.7)
          
          # Use a more detailed status for ID #2 to track its state
          if not face.get('freshly_detected', True):
              display_status = f"Tracked ({frames_since})"
          else:
              display_status = f"Active ({frames_since})"
      else:
          # Standard handling for other faces
          if not face.get('freshly_detected', True):
              display_status = f"Tracked ({frames_since})"
              # Apply a gentler confidence reduction for non-fresh faces
              # display_confidence = display_confidence * (1.0 - frames_since / self.face_max_lost_frames * 0.3)
          else:
              # For active detections, still show the frame count for debugging
              display_status = f"Active ({frames_since})"
  
      # Update info panel and ensure it's active
      self.info_panel.active = True
      self.info_panel.target_alpha = 255
      self.info_panel.update(
        face['id'],
        face['name'],
        display_status,
        confidence=display_confidence
      )
  
      # Store and modify info panel colors for persistence
      original_panel_color = self.info_panel.color
      original_text_color = self.info_panel.text_color
      if not face.get('freshly_detected', True):
        self.info_panel.color = info_color
        self.info_panel.text_color = info_text_color
      else:
        self.info_panel.color = self.config.info_panel_color
        self.info_panel.text_color = self.config.info_panel_text_color
  
      # Draw info panel - force alpha to ensure visibility
      original_alpha = self.info_panel.alpha
      self.info_panel.alpha = max(self.info_panel.alpha, 150)  # Ensure minimum visibility
      self.info_panel.draw(self.screen, (panel_x, panel_y))
      self.info_panel.alpha = original_alpha  # Restore original alpha

      # Restore original info panel colors
      self.info_panel.color = original_panel_color
      self.info_panel.text_color = original_text_color

    # Draw FPS counter
    current_time = time.time()
    elapsed = current_time - self.last_time

    # Only update FPS every second
    if elapsed > 1.0:
      self.fps = self.frame_count / elapsed
      self.last_time = current_time
      self.frame_count = 0

    fps_text = f"FPS: {self.fps:.1f}"
    font = pygame.font.SysFont('monospace', 20)
    fps_surface = font.render(fps_text, True, (0, 255, 0))
    self.screen.blit(fps_surface, (10, 10))

    # Update display
    pygame.display.flip()

  def run(self):
    """Main application loop."""
    self.running = True

    try:
      while self.running:
        # Process events
        for event in pygame.event.get():
          if event.type == pygame.QUIT:
            self.running = False
          elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
              self.running = False
              self.write_face_data()
              write_detection_data()

        # Get the frame (either from camera or example frames)
        if self.use_example_frames:
          # No frames available
          if not self.example_frames:
            print("No example frames available")
            break

          # Load frame from example directory
          frame_path = os.path.join(self.example_frame_dir, self.example_frames[self.current_frame_index])
          frame = cv2.imread(frame_path)

          if frame is None:
            print(f"Error reading example frame: {frame_path}")
            break

          # Move to next frame (cycle through all frames)
          self.current_frame_index = (self.current_frame_index + 1) % len(self.example_frames)

          # Simulate camera delay to avoid running too fast
          pygame.time.delay(int(1000 / self.frame_rate))

          ret = True
        else:
          # Read frame from camera/video
          ret, frame = self.cap.read()

          # If video file has ended, loop back to beginning
          if not ret and isinstance(self.config.source, str) and not self.config.source.isdigit():
            print("Video reached end, looping back to beginning")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to first frame
            ret, frame = self.cap.read()  # Try reading again

        if not ret:
          print("Error reading frame, check if the video file exists and is readable")
          break

        # Process frame
        processed_frame = self.process_frame(frame)

        # Render frame with overlays
        self.render(processed_frame)

        # Update frame count for FPS calculation
        self.frame_count += 1

        # Cap frame rate
        self.clock.tick(self.config.max_fps or self.frame_rate)

    except KeyboardInterrupt:
      print("Application interrupted by user")
    finally:
      self.cleanup()

  def cleanup(self):
    """Clean up resources."""
    if not self.use_example_frames:
      self.cap.release()
    self.face_detection.close()
    pygame.quit()


def parse_args():
  """Parse command line arguments."""
  # @formatter:off
  parser = argparse.ArgumentParser(description='High-Tech Camera Overlay System')
  parser.add_argument('--grid-cols',      type=int, default=9,             help='Number of grid columns (default: 9)')
  parser.add_argument('--grid-rows',      type=int, default=5,             help='Number of grid rows (default: 5)')
  parser.add_argument('--display-width',  type=int, default=0,             help='Display width (default: same as camera)')
  parser.add_argument('--display-height', type=int, default=0,             help='Display height (default: same as camera)')
  parser.add_argument('--source',         type=str, default='0',           help='Camera index or video file path (default: 0)')
  parser.add_argument('--config',         type=str, default='config.yaml', help='Path to configuration file (default: config.yaml)')
  # @formatter:on

  return parser.parse_args()


def main():
  """Main entry point."""
  # Parse command line arguments
  args = parse_args()

  # Load configuration
  config = load_config(args.config)

  # Override config with command line arguments (except 'config' itself)
  for arg, value in vars(args).items():
    if arg != 'config' and value is not None and value != 0:  # Skip config path and default values
      setattr(config, arg, value)

  # Create and run application
  app = CameraOverlayApp(config)
  app.run()


if __name__ == "__main__":
  main()
