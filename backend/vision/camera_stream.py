import cv2
import time
import threading

class CameraStream:
    """
    Handles streaming video from a smartphone camera via IP Webcam/DroidCam,
    or falls back to a standard webcam (0).
    Runs reading in a separate thread to avoid blocking and lagging.
    """
    def __init__(self, src=0):
        # src can be 0 (webcam) or an IP string like 'http://192.168.1.100:8080/video'
        self.src = src
        self.stream = cv2.VideoCapture(src)
        self.stopped = False
        self.grabbed = False
        self.frame = None
        self.thread = None
        
        if self.stream.isOpened():
            self.grabbed, self.frame = self.stream.read()
            # Start background thread to read frames continuously
            self.thread = threading.Thread(target=self.update, args=())
            self.thread.daemon = True
            self.thread.start()
        else:
            print(f"Warning: Could not open camera source {src}")

    def update(self):
        # Keep looping infinitely until the thread is stopped
        while True:
            if self.stopped:
                return
            
            # Read the next frame from the stream
            grabbed, frame = self.stream.read()
            if grabbed:
                self.grabbed = grabbed
                self.frame = frame
            time.sleep(0.01) # Small sleep to not hog CPU

    def read(self):
        # Return the frame most recently read
        return self.frame

    def stop(self):
        # Indicate that the thread should be stopped
        self.stopped = True
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.stream is not None:
            self.stream.release()
