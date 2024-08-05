import os
import time
import datetime
import logging
import cv2
import pyrealsense2 as rs
import numpy as np
from pathlib import Path
from typing import Tuple
logging.getLogger().setLevel(logging.INFO)

class CameraD435i:
    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        _width: int = 640,
        _height: int = 480,
        frame_rate: int = 30,
        window_title: str = "CameraD435i",
        save_path: str = "record",
        stream: bool = False,
        save: bool = False,
        log: bool = True,) -> None:
            self.width = width # width of the display window
            self.height = height # height of the display window
            self._width = _width # width of the camera frame
            self._height = _height # height of the camera frame
            self.frame_rate = frame_rate
            self.window_title = window_title
            self.save_path = Path(save_path)
            self.stream = stream
            self.save = save
            self.log = log
            
            # Make record directory
            if save:
                assert save_path is not None, "Please provide a save path"
                os.makedirs(self.save_path, exist_ok=True) # if path does not exist, create it
                self.save_path = self.save_path / f'{len(os.listdir(self.save_path)) + 1:06d}'
                os.makedirs(self.save_path, exist_ok=True)
                
                logging.info(f"Save directory: {self.save_path}") 
            
            # config and init stream pipeline
            self.pipeline, self.cfg = self.realsense_pipeline()
            self.pipeline.start(self.cfg)

    def realsense_pipeline(self):
        """
        Return a pipeline for capturing from the realsense D435i camera
        """
        # Configure depth and color streams
        pipeline = rs.pipeline()
        config = rs.config()
        
        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))
        print("Device product line: ", device_product_line)
        
        device_sensors = str(device.query_sensors())
        print("Device sensors: ", device_sensors) # D435i: [Stereo Module, RGB Camera, Motion Module]
        
        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)

        config.enable_stream(rs.stream.depth, self._width, self._height, rs.format.z16, self.frame_rate)
        config.enable_stream(rs.stream.color, self._width, self._height, rs.format.bgr8, self.frame_rate)
        
        return pipeline, config

    def run(self) -> None:
        """
        Streaming camera feed
        """
        
        # Start streaming
        self.pipeline.start(self.cfg)

        try:
            while True:
                t0 = time.time()
                timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
                
                # Wait for a coherent pair of frames: depth and color
                frames = self.pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                # Convert images to numpy arrays
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

                depth_colormap_dim = depth_colormap.shape
                color_colormap_dim = color_image.shape

                # If depth and color resolutions are different, resize color image to match depth image for display
                if depth_colormap_dim != color_colormap_dim:
                    resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
                    images = np.hstack((resized_color_image, depth_colormap))
                else:
                    images = np.hstack((color_image, depth_colormap))
                
                if self.log:
                    print(f"FPS: {1 / (time.time() - t0):.2f}")

                if self.save: 
                    cv2.imwrite(str(self.save_path / f"{timestamp}.jpg"), images)
                
                if self.stream:    
                    # Show images
                    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                    cv2.imshow('RealSense', images)
                    key = cv2.waitKey(1)
                    if key == ord('q') or key == 27: # press q or esc to quit
                        break
                
        except Exception as e:
            print(e)
            
        finally:

            # Stop streaming
            self.pipeline.stop()
            cv2.destroyAllWindows()       

    @property
    def frame(self) -> Tuple[np.ndarray, np.ndarray]:
        
        while True:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            return color_image, depth_image

        
        
        
        
if __name__ == '__main__':
    cam = CameraD435i(stream=True)
    cam.run()

