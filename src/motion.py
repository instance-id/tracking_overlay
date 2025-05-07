#!/usr/bin/env -S uv run

# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "opencv-python>=4.8.0",
#   "numpy>=1.24.0",
#   "imutils>=0.5.4"
# ]
# ///

"""
Motion detection module for high-tech camera overlay system.
Handles detecting and analyzing motion in video frames.
"""

import cv2
import numpy as np
import imutils
from datetime import datetime


class MotionDetector:
    """Motion detector for identifying movement in video frames."""
    
    def __init__(self, grid_cols=9, grid_rows=5, frame_width=1280, frame_height=720,
                 sensitivity=0.3, blur=5, threshold=25):
        """
        Initialize motion detector.
        
        Args:
            grid_cols: Number of grid columns for motion detection zones
            grid_rows: Number of grid rows for motion detection zones
            frame_width: Width of video frames
            frame_height: Height of video frames
            sensitivity: Motion sensitivity (0.0 - 1.0)
            blur: Gaussian blur size for noise reduction
            threshold: Threshold for motion detection
        """
        self.grid_cols = grid_cols
        self.grid_rows = grid_rows
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.sensitivity = sensitivity
        self.blur = blur
        self.threshold = threshold
        
        # Calculate grid cell dimensions
        self.cell_width = frame_width // grid_cols
        self.cell_height = frame_height // grid_rows
        
        # Initialize background model with sensitivity-aware parameters
        # Higher sensitivity (0.0-1.0) means less sensitive motion detection
        # So we increase varThreshold and history when sensitivity is high
        bg_history = int(500 + 500 * sensitivity)  # 500-1000 based on sensitivity
        bg_threshold = int(16 + 40 * sensitivity)   # 16-56 based on sensitivity
        
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=bg_history,
            varThreshold=bg_threshold,
            detectShadows=False
        )
        
        # Previous frame storage
        self.prev_frame = None
        self.frame_delta = None
        self.threshold_image = None
        
        # Motion grid (stores motion level for each cell)
        self.motion_grid = np.zeros((grid_rows, grid_cols), dtype=np.float32)
        
        # Motion history (for temporal smoothing)
        self.motion_history = np.zeros((grid_rows, grid_cols, 5), dtype=np.float32)
        self.history_index = 0
        
        # Timestamp for performance tracking
        self.last_process_time = datetime.now()
        
    def detect(self, frame):
        """
        Detect motion in the provided frame.
        
        Args:
            frame: OpenCV image frame
            
        Returns:
            2D numpy array of shape (rows, cols) with motion intensity values (0.0 - 1.0)
        """
        # Record processing start time
        start_time = datetime.now()
        
        # Preprocess frame
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply gaussian blur
        gray = cv2.GaussianBlur(gray, (self.blur, self.blur), 0)
        
        # Apply background subtraction
        mask = self.bg_subtractor.apply(gray)
        
        # Apply dynamic thresholding based on sensitivity setting
        # Higher sensitivity value means we need a higher threshold to filter out more noise
        dynamic_threshold = int(self.threshold * (1 + self.sensitivity))
        thresh = cv2.threshold(mask, dynamic_threshold, 255, cv2.THRESH_BINARY)[1]
        
        # For high sensitivity, apply more aggressive morphological operations
        if self.sensitivity > 0.5:
            # First erode to remove small noise
            kernel_size = max(1, int(3 * self.sensitivity))
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            thresh = cv2.erode(thresh, kernel, iterations=1)
            
            # Then dilate, but with fewer iterations when sensitivity is high
            iterations = max(1, 3 - int(2 * self.sensitivity))
            thresh = cv2.dilate(thresh, kernel, iterations=iterations)
        else:
            # Standard dilation for normal sensitivity
            thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Store processed frame for debugging
        self.threshold_image = thresh
        
        # Calculate motion for each grid cell
        self._calculate_grid_motion(thresh)
        
        # Apply temporal smoothing
        self._apply_temporal_smoothing()
        
        # Store motion processing time
        self.last_process_time = datetime.now() - start_time
        
        # Return the motion grid
        return self.motion_grid
        
    def _calculate_grid_motion(self, thresh_image):
        """
        Calculate motion level for each grid cell.
        
        Args:
            thresh_image: Thresholded binary image
        """
        # Reset motion grid
        self.motion_grid.fill(0)
        
        # Process each grid cell
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                # Calculate cell ROI
                x1 = col * self.cell_width
                y1 = row * self.cell_height
                x2 = x1 + self.cell_width
                y2 = y1 + self.cell_height
                
                # Extract cell ROI
                cell_roi = thresh_image[y1:y2, x1:x2]
                
                # Calculate percentage of motion pixels in cell
                if cell_roi.size > 0:
                    motion_pixels = np.count_nonzero(cell_roi)
                    total_pixels = cell_roi.size
                    motion_ratio = motion_pixels / total_pixels
                    
                    # Apply more aggressive sensitivity scaling
                    # Higher sensitivity value (0.0-1.0) means LESS sensitive
                    # We'll use a power function to make small motions much less likely to trigger
                    # when sensitivity is high
                    scaled_ratio = motion_ratio * (1.0 - self.sensitivity)
                    
                    # Apply additional thresholding for more aggressive filtering
                    min_threshold = 0.01 * self.sensitivity
                    if scaled_ratio < min_threshold:
                        scaled_ratio = 0.0
                    
                    # Update motion grid
                    self.motion_grid[row, col] = scaled_ratio
                    
    def _apply_temporal_smoothing(self):
        """Apply temporal smoothing to reduce noise in motion detection."""
        # Add current motion to history
        self.motion_history[:, :, self.history_index] = self.motion_grid
        
        # Update history index
        self.history_index = (self.history_index + 1) % self.motion_history.shape[2]
        
        # Calculate weighted average (more recent frames have higher weight)
        weights = np.array([0.1, 0.15, 0.2, 0.25, 0.3])
        
        # Reorder weights based on current history index
        indices = np.roll(np.arange(5), -self.history_index)
        sorted_weights = weights[indices]
        
        # Apply weighted average
        avg_motion = np.average(self.motion_history, axis=2, weights=sorted_weights)
        
        # Apply additional sensitivity-based filtering for temporal smoothing
        # When sensitivity is high, require more consistent motion over time
        if self.sensitivity > 0.5:
            # Calculate the variance across time for each cell
            variance = np.var(self.motion_history, axis=2)
            # If variance is high (inconsistent motion), reduce the average motion
            variance_threshold = 0.01 * (self.sensitivity - 0.5) * 2.0  # scale with sensitivity
            inconsistent_mask = variance > variance_threshold
            avg_motion[inconsistent_mask] *= (1.0 - self.sensitivity * 0.5)
        
        # Additional threshold for high sensitivity settings
        if self.sensitivity > 0.7:
            # For very high sensitivity, apply stronger minimum threshold
            high_sens_threshold = (self.sensitivity - 0.7) * 0.2
            avg_motion[avg_motion < high_sens_threshold] = 0.0
            
        self.motion_grid = avg_motion
        
    def get_motion_areas(self, threshold=0.5):
        """
        Get list of grid cells with motion above threshold.
        
        Args:
            threshold: Motion threshold (0.0 - 1.0)
            
        Returns:
            List of (row, col) tuples for cells with motion
        """
        motion_areas = []
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                if self.motion_grid[row, col] > threshold:
                    motion_areas.append((row, col))
        
        return motion_areas
        
    def get_debug_frame(self):
        """
        Get debug visualization frame.
        
        Returns:
            OpenCV frame with motion visualization
        """
        # Create empty frame
        debug_frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        
        # Draw grid
        for row in range(self.grid_rows + 1):
            y = row * self.cell_height
            cv2.line(debug_frame, (0, y), (self.frame_width, y), (50, 50, 50), 1)
            
        for col in range(self.grid_cols + 1):
            x = col * self.cell_width
            cv2.line(debug_frame, (x, 0), (x, self.frame_height), (50, 50, 50), 1)
            
        # Draw motion levels
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                # Calculate cell ROI
                x1 = col * self.cell_width
                y1 = row * self.cell_height
                x2 = x1 + self.cell_width
                y2 = y1 + self.cell_height
                
                # Get motion level
                motion = self.motion_grid[row, col]
                
                # Skip cells with no motion
                if motion <= 0.05:
                    continue
                
                # Calculate color based on motion (green to red)
                g = int(255 * (1 - motion))
                r = int(255 * motion)
                color = (0, g, r)
                
                # Fill cell with semi-transparent color
                cv2.rectangle(
                    debug_frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    -1  # Filled
                )
                
                # Draw motion text
                text = f"{motion:.2f}"
                cv2.putText(
                    debug_frame,
                    text,
                    (x1 + 5, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1
                )
        
        # Add threshold image if available
        if self.threshold_image is not None:
            # Create a three-channel version
            thresh_color = cv2.cvtColor(self.threshold_image, cv2.COLOR_GRAY2BGR)
            
            # Resize to fit in a corner
            display_width = self.frame_width // 4
            display_height = self.frame_height // 4
            thresh_display = cv2.resize(thresh_color, (display_width, display_height))
            
            # Overlay in the corner
            debug_frame[0:display_height, 0:display_width] = thresh_display
        
        return debug_frame
        
    def reset(self):
        """Reset motion detector state."""
        # Reset background model
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=False
        )
        
        # Reset motion grid and history
        self.motion_grid.fill(0)
        self.motion_history.fill(0)
        self.history_index = 0


class ONVIFMotionProvider:
    """ONVIF motion detection provider for external camera feeds."""
    
    def __init__(self, camera_url, username, password, grid_cols=9, grid_rows=5):
        """
        Initialize ONVIF motion provider.
        
        Args:
            camera_url: URL of ONVIF camera
            username: ONVIF camera username
            password: ONVIF camera password
            grid_cols: Number of grid columns
            grid_rows: Number of grid rows
        """
        self.camera_url = camera_url
        self.username = username
        self.password = password
        self.grid_cols = grid_cols
        self.grid_rows = grid_rows
        
        # Motion grid
        self.motion_grid = np.zeros((grid_rows, grid_cols), dtype=np.float32)
        
        # Connection status
        self.connected = False
        
    def connect(self):
        """
        Connect to ONVIF camera.
        
        Returns:
            bool: True if connected successfully
        """
        # Placeholder for actual ONVIF implementation
        # In a real implementation, this would use zeep or onvif library
        print(f"Connecting to ONVIF camera at {self.camera_url}")
        self.connected = True
        return self.connected
        
    def get_motion_grid(self):
        """
        Get motion grid from ONVIF camera.
        
        Returns:
            2D numpy array of shape (rows, cols) with motion intensity values (0.0 - 1.0)
        """
        # Placeholder for actual ONVIF implementation
        # In a real implementation, this would fetch analytics from the camera
        
        # Simulate some motion for testing
        if not self.connected:
            return np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)
            
        # Clear previous motion
        self.motion_grid.fill(0)
        
        # Simulate random motion (for testing only)
        # In a real implementation, this would be replaced with actual ONVIF data
        for _ in range(3):  # Simulate a few active cells
            row = np.random.randint(0, self.grid_rows)
            col = np.random.randint(0, self.grid_cols)
            self.motion_grid[row, col] = np.random.uniform(0.5, 1.0)
            
        return self.motion_grid


if __name__ == "__main__":
    # Simple test code
    import time
    import matplotlib.pyplot as plt
    
    # Create test frame
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Create motion detector
    detector = MotionDetector(grid_cols=8, grid_rows=6, frame_width=640, frame_height=480)
    
    # Create figure for visualization
    plt.figure(figsize=(10, 8))
    
    # Process frames
    for i in range(100):
        # Create test frame with moving object
        test_frame.fill(0)
        
        # Draw moving object
        x = int((i % 10) * 64)
        y = int((i // 10) * 80)
        cv2.rectangle(test_frame, (x, y), (x + 50, y + 50), (0, 0, 255), -1)
        
        # Detect motion
        motion_grid = detector.detect(test_frame)
        
        # Get debug frame
        debug_frame = detector.get_debug_frame()
        
        # Display results
        plt.clf()
        plt.subplot(2, 1, 1)
        plt.imshow(cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB))
        plt.title("Input Frame")
        
        plt.subplot(2, 1, 2)
        plt.imshow(cv2.cvtColor(debug_frame, cv2.COLOR_BGR2RGB))
        plt.title("Motion Detection")
        
        plt.tight_layout()
        plt.pause(0.1)
    
    plt.close()
