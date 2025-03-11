# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Sequence, Literal
import logging
import os
import time
import datetime
from pathlib import Path
import cv2
import numpy as np
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

logging.getLogger().setLevel(logging.INFO)

class Camera:
    def __init__(
        self,
        sensor_id: int | Sequence[int] = 0,
        camera_type: Literal["csi", "usb"] = "csi",  # New parameter for camera type
        width: int = 1920,          # Display window width
        height: int = 1080,         # Display window height
        capture_width: int = 640,   # Actual capture width from camera
        capture_height: int = 480, # Actual capture height from camera
        frame_rate: int = 30,      # Frame rate (fps)
        flip_method: int = 0,      # Flip method (0 for none, 2 for 180�� flip)
        window_title: str = "Camera",
        save_path: str = "record",
        stream: bool = False,       # Enable real-time streaming
        save: bool = False,         # Enable frame saving
        log: bool = True,           # Enable performance logging
    ) -> None:
        """
        Initialize camera pipeline for CSI or USB devices.
        
        Args:
            sensor_id: Camera device ID (sensor ID for CSI, /dev/videoX number for USB)
            camera_type: 'csi' for NVIDIA CSI cameras, 'usb' for standard USB cameras
        """
        self.camera_type = camera_type.lower()
        self.sensor_id = sensor_id
        self.width = width
        self.height = height
        self.capture_width = capture_width
        self.capture_height = capture_height
        self.frame_rate = frame_rate
        self.flip_method = flip_method
        self.window_title = window_title
        self.save_path = Path(save_path)
        self.stream = stream
        self.save = save
        self.log = log
        
        # Validate camera type
        if self.camera_type not in ["csi", "usb"]:
            raise ValueError("camera_type must be 'csi' or 'usb'")
        
        # Convert sensor_id to list
        if isinstance(sensor_id, int):
            self.sensor_id = [sensor_id]
        elif isinstance(sensor_id, Sequence):
            self.sensor_id = list(sensor_id)
            
        # Check device existence
        for sid in self.sensor_id:
            if self.camera_type == "usb":
                dev_path = Path(f"/dev/video{sid}")
                if not dev_path.exists():
                    raise FileNotFoundError(f"USB camera /dev/video{sid} not detected")

        # Initialize capture pipelines
        self.caps = [
            cv2.VideoCapture(
                self._generate_gstreamer_pipeline(sid),
                cv2.CAP_GSTREAMER
            ) for sid in self.sensor_id
        ]
        
        # Create save directory
        if self.save:
            os.makedirs(self.save_path, exist_ok=True)
            self.save_path = self.save_path / f'{len(os.listdir(self.save_path)) + 1:06d}'
            os.makedirs(self.save_path, exist_ok=True)
            logging.info(f"Save directory: {self.save_path}")

    def _generate_gstreamer_pipeline(self, sensor_id: int) -> str:
        """Generate GStreamer pipeline string based on camera type"""
        if self.camera_type == "csi":
            # NVIDIA CSI camera pipeline with hardware acceleration
            return (
                "nvarguscamerasrc sensor-id=%d ! "
                "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
                "nvvidconv flip-method=%d ! "
                "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
                "videoconvert ! video/x-raw, format=(string)BGR ! appsink"
                % (
                    sensor_id,
                    self.capture_width,
                    self.capture_height,
                    self.frame_rate,
                    self.flip_method,
                    self.width,
                    self.height,
                )
            )
        else:  # USB camera
            # Standard UVC camera pipeline
            return (
                f"v4l2src device=/dev/video{sensor_id} ! "
                "video/x-raw, "
                f"width={self.capture_width}, height={self.capture_height}, framerate={self.frame_rate}/1 ! "
                "videoconvert ! video/x-raw, format=BGR ! appsink"
            )

    def run(self) -> None:
        """Main capture loop"""
        try:
            while True:
                t_start = time.time()
                
                # Initialize frames list
                frames = []
                
                # Read frames with validity check
                for i, cap in enumerate(self.caps):
                    ret, frame = cap.read()
                    if not ret:
                        raise RuntimeError(f"Failed to read frame from camera {i} (ID={self.sensor_id[i]})")
                    if frame is None or frame.size == 0:
                        raise ValueError(f"Empty frame detected from camera {i}")
                    frames.append(frame)
                    if self.log:
                        print(f"Camera {i} frame size: {frame.shape[1]}x{frame.shape[0]}")
                
                # Save frames
                if self.save:
                    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
                    for i, frame in enumerate(frames):
                        cv2.imwrite(str(self.save_path / f"cam{i}_{timestamp}.jpg"), frame)
                
                # Display frames
                if self.stream:
                    for i, frame in enumerate(frames):
                        cv2.imshow(f'{self.window_title} #{i}', frame)
                    if cv2.waitKey(1) == ord('q'):
                        break
                
                # Log performance
                if self.log:
                    fps = 1 / (time.time() - t_start)
                    print(f"FPS: {fps:.2f} | Active cameras: {len(self.caps)}")
                
        except KeyboardInterrupt:
            logging.info("User interrupted")
        finally:
            for cap in self.caps:
                if cap.isOpened():
                    cap.release()  # 释放 VideoCapture 对象
            cv2.destroyAllWindows()
            # 显式设置 GStreamer 管道为 NULL 状态
            for cap in self.caps:
                if hasattr(cap, 'get_pipeline'):
                    pipeline = cap.get_pipeline()
                    if pipeline:
                        pipeline.set_state(Gst.State.NULL)

    @property
    def frame(self) -> list[np.ndarray]:
        """Get current frames from all cameras"""
        return [cap.read()[1] for cap in self.caps]

if __name__ == '__main__':
    # Example for CSI camera
    # cam = Camera(camera_type='csi', sensor_id=0, stream=True)
    
    # Example for USB camera (modify sensor_id as needed)
    cam = Camera(
        camera_type='usb',
        sensor_id=0,
        capture_width=640,
        capture_height=480,
        frame_rate=30,
        stream=True
    )
    cam.run()