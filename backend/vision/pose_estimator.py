import cv2
import mediapipe as mp
import numpy as np


class PoseEstimator:
    """
    Wraps MediaPipe BlazePose for ACL-focused lower-body tracking.
    - model_complexity=2  (most accurate, still real-time on M-series Mac)
    - smooth_landmarks=True  (MediaPipe's own Kalman-like filter – always on)
    - We do NOT apply a secondary EMA here; MediaPipe's smoother is sufficient
      and adding our own alpha inverted the smoothing direction in the old code.
    - Visibility threshold kept at 0.5 so required hip/knee/ankle landmarks
      are never silently skipped (0.7 was too aggressive and caused KeyErrors).
    """

    # ACL-relevant joints that must be present for a valid frame
    REQUIRED = {
        'LEFT_HIP', 'RIGHT_HIP',
        'LEFT_KNEE', 'RIGHT_KNEE',
        'LEFT_ANKLE', 'RIGHT_ANKLE',
        'LEFT_SHOULDER', 'RIGHT_SHOULDER',
    }

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,          # Highest accuracy
            smooth_landmarks=True,       # MediaPipe's built-in temporal smoother
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

        # Only draw ACL-demo connections: torso reference, hips, knees, ankles, feet.
        self.acl_connections = frozenset([
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER,   self.mp_pose.PoseLandmark.RIGHT_SHOULDER),
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER,   self.mp_pose.PoseLandmark.LEFT_HIP),
            (self.mp_pose.PoseLandmark.RIGHT_SHOULDER,  self.mp_pose.PoseLandmark.RIGHT_HIP),
            (self.mp_pose.PoseLandmark.LEFT_HIP,        self.mp_pose.PoseLandmark.RIGHT_HIP),
            (self.mp_pose.PoseLandmark.LEFT_HIP,        self.mp_pose.PoseLandmark.LEFT_KNEE),
            (self.mp_pose.PoseLandmark.RIGHT_HIP,       self.mp_pose.PoseLandmark.RIGHT_KNEE),
            (self.mp_pose.PoseLandmark.LEFT_KNEE,       self.mp_pose.PoseLandmark.LEFT_ANKLE),
            (self.mp_pose.PoseLandmark.RIGHT_KNEE,      self.mp_pose.PoseLandmark.RIGHT_ANKLE),
            (self.mp_pose.PoseLandmark.LEFT_ANKLE,      self.mp_pose.PoseLandmark.LEFT_HEEL),
            (self.mp_pose.PoseLandmark.RIGHT_ANKLE,     self.mp_pose.PoseLandmark.RIGHT_HEEL),
            (self.mp_pose.PoseLandmark.LEFT_HEEL,       self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX),
            (self.mp_pose.PoseLandmark.RIGHT_HEEL,      self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX),
            (self.mp_pose.PoseLandmark.LEFT_ANKLE,      self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX),
            (self.mp_pose.PoseLandmark.RIGHT_ANKLE,     self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX),
        ])

        self.connection_spec = self.mp_drawing.DrawingSpec(
            color=(0, 200, 255), thickness=3
        )
        self.landmark_spec = self.mp_drawing.DrawingSpec(
            color=(255, 255, 0), thickness=-1, circle_radius=5
        )

        self.acl_landmarks = {
            landmark
            for connection in self.acl_connections
            for landmark in connection
        }

    def process_frame(self, frame):
        """Convert BGR→RGB and run MediaPipe inference."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        return self.pose.process(rgb)

    def extract_landmarks(self, results):
        """
        Return a flat dict of landmark_name → {x,y,z,visibility}.
        Returns None if any ACL-required joint is missing or low-visibility.
        """
        if not results.pose_landmarks:
            return None

        raw = {}
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            name = self.mp_pose.PoseLandmark(idx).name
            raw[name] = {
                'x': lm.x,
                'y': lm.y,
                'z': lm.z,
                'visibility': lm.visibility,
            }

        # Gate: ALL required joints must be visible (threshold 0.5)
        for key in self.REQUIRED:
            if key not in raw or raw[key]['visibility'] < 0.5:
                return None

        return raw

    def draw_landmarks(self, frame, results, highlight_joints=None):
        """
        Draw lower-body ACL skeleton on frame.
        highlight_joints: dict of joint_name → BGR color tuple for XAI overlays.
        """
        out = frame.copy()

        if not results.pose_landmarks:
            return out

        h, w = out.shape[:2]
        landmarks = results.pose_landmarks.landmark

        def point_for(landmark_id):
            lm = landmarks[landmark_id.value]
            if lm.visibility < 0.45:
                return None
            x = int(np.clip(lm.x, 0.0, 1.0) * w)
            y = int(np.clip(lm.y, 0.0, 1.0) * h)
            return x, y

        # Step 1 — draw only the custom ACL skeleton. MediaPipe's helper draws
        # every landmark dot even with filtered connections, so this stays clean.
        for start, end in self.acl_connections:
            p1 = point_for(start)
            p2 = point_for(end)
            if p1 and p2:
                cv2.line(out, p1, p2, self.connection_spec.color, self.connection_spec.thickness)

        for landmark_id in self.acl_landmarks:
            p = point_for(landmark_id)
            if p:
                cv2.circle(out, p, self.landmark_spec.circle_radius, self.landmark_spec.color, -1)
                cv2.circle(out, p, self.landmark_spec.circle_radius + 2, (15, 23, 42), 1)

        # Step 2 — XAI joint highlights
        if highlight_joints:
            for joint_name, color in highlight_joints.items():
                try:
                    idx = getattr(self.mp_pose.PoseLandmark, joint_name)
                    lm = landmarks[idx.value]
                    if lm.visibility > 0.5:
                        cx, cy = int(np.clip(lm.x, 0.0, 1.0) * w), int(np.clip(lm.y, 0.0, 1.0) * h)
                        cv2.circle(out, (cx, cy), 14, color, -1)
                        cv2.circle(out, (cx, cy), 18, (255, 255, 255), 2)
                except AttributeError:
                    pass

        return out
