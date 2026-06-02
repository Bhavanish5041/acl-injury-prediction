import numpy as np
import time
from collections import deque


class BiomechanicsExtractor:
    """
    Extracts ACL-relevant biomechanical features from MediaPipe 3D landmarks.

    Fixes applied vs previous version:
    - Removed erroneous `networkx` import (was causing ImportError crash)
    - Fixed broken indentation on right_knee_flexion np.clip block
    - Knee valgus is now calculated with proper frontal-plane vector math,
      NOT a raw x-coordinate difference (which gave garbage scaled values)
    - Joint velocity is low-pass filtered (rolling mean over 5 frames) to
      prevent single-frame spikes from spiking the risk score
    - All outputs are clipped to physiologically plausible ranges
    """

    def __init__(self):
        self.prev_landmarks = None
        self.prev_time = None
        # Keep a small window of recent velocities to smooth spikes
        self._velocity_buffer = deque(maxlen=5)

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _vec3(self, a, b):
        """Return 3-D vector from landmark b to landmark a."""
        return np.array([a['x'] - b['x'], a['y'] - b['y'], a['z'] - b['z']])

    def angle_between(self, p1, vertex, p2):
        """
        Angle at `vertex` formed by the rays vertex→p1 and vertex→p2.
        Returns degrees in [0, 180].
        """
        v1 = self._vec3(p1, vertex)
        v2 = self._vec3(p2, vertex)
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return 0.0
        cos_a = np.dot(v1, v2) / (n1 * n2)
        return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))

    def knee_valgus_frontal(self, hip, knee, ankle):
        """
        Approximate knee valgus in the frontal (X-Y) plane.
        We project hip→knee and knee→ankle onto the X-Y plane and measure
        the angle between them.  Deviation from 180° (straight leg) is the
        valgus/varus magnitude.  Clipped to [0, 30] degrees.
        """
        v1 = np.array([hip['x'] - knee['x'], hip['y'] - knee['y']])
        v2 = np.array([ankle['x'] - knee['x'], ankle['y'] - knee['y']])
        n1 = np.linalg.norm(v1) + 1e-6
        n2 = np.linalg.norm(v2) + 1e-6
        cos_a = np.dot(v1 / n1, v2 / n2)
        angle_deg = float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))
        deviation = abs(180.0 - angle_deg)          # 0 = straight, higher = more valgus
        return float(np.clip(deviation, 0.0, 30.0))

    # ------------------------------------------------------------------
    # Velocity
    # ------------------------------------------------------------------

    def _joint_velocity(self, landmarks, current_time):
        if self.prev_landmarks is None or self.prev_time is None:
            return 0.0
        dt = current_time - self.prev_time
        if dt <= 0:
            return 0.0
        # Track centre-of-mass proxy: midpoint of hips
        curr = np.array([
            (landmarks['LEFT_HIP']['x'] + landmarks['RIGHT_HIP']['x']) / 2,
            (landmarks['LEFT_HIP']['y'] + landmarks['RIGHT_HIP']['y']) / 2,
        ])
        prev = np.array([
            (self.prev_landmarks['LEFT_HIP']['x'] + self.prev_landmarks['RIGHT_HIP']['x']) / 2,
            (self.prev_landmarks['LEFT_HIP']['y'] + self.prev_landmarks['RIGHT_HIP']['y']) / 2,
        ])
        raw_velocity = float(np.linalg.norm(curr - prev) / dt)
        self._velocity_buffer.append(raw_velocity)
        return float(np.mean(self._velocity_buffer))   # smoothed

    # ------------------------------------------------------------------
    # Main extraction
    # ------------------------------------------------------------------

    def extract_features(self, landmarks):
        """
        Takes the landmark dict from PoseEstimator.extract_landmarks() and
        returns a nested dict:
          { 'model_input': { feature_name: float, ... } }

        Returns None if any required landmark is missing.
        """
        if not landmarks:
            return None

        current_time = time.time()

        try:
            L_HIP = landmarks['LEFT_HIP']
            R_HIP = landmarks['RIGHT_HIP']
            L_KNEE = landmarks['LEFT_KNEE']
            R_KNEE = landmarks['RIGHT_KNEE']
            L_ANK = landmarks['LEFT_ANKLE']
            R_ANK = landmarks['RIGHT_ANKLE']
            L_SHO = landmarks['LEFT_SHOULDER']
            R_SHO = landmarks['RIGHT_SHOULDER']
        except KeyError:
            return None

        # 1. Knee flexion  (straight = 180°, so flexion = 180 - raw_angle)
        left_knee_flexion  = float(np.clip(180.0 - self.angle_between(L_HIP, L_KNEE, L_ANK),  0, 150))
        right_knee_flexion = float(np.clip(180.0 - self.angle_between(R_HIP, R_KNEE, R_ANK), 0, 150))

        # 2. Knee valgus  (frontal plane deviation from straight)
        left_knee_valgus  = self.knee_valgus_frontal(L_HIP, L_KNEE, L_ANK)
        right_knee_valgus = self.knee_valgus_frontal(R_HIP, R_KNEE, R_ANK)

        # 3. Hip flexion  (Shoulder → Hip → Knee, same convention)
        left_hip_flexion  = float(np.clip(180.0 - self.angle_between(L_SHO, L_HIP, L_KNEE),  0, 120))
        right_hip_flexion = float(np.clip(180.0 - self.angle_between(R_SHO, R_HIP, R_KNEE), 0, 120))

        # 4. Asymmetry scores
        knee_asymmetry = float(abs(left_knee_flexion - right_knee_flexion))
        hip_asymmetry  = float(abs(left_hip_flexion  - right_hip_flexion))

        # 5. Smoothed joint velocity
        joint_velocity = self._joint_velocity(landmarks, current_time)

        # Update history
        self.prev_landmarks = landmarks
        self.prev_time = current_time

        return {
            'model_input': {
                'left_knee_flexion':  left_knee_flexion,
                'right_knee_flexion': right_knee_flexion,
                'left_knee_valgus':   left_knee_valgus,
                'right_knee_valgus':  right_knee_valgus,
                'left_hip_flexion':   left_hip_flexion,
                'right_hip_flexion':  right_hip_flexion,
                'knee_asymmetry':     knee_asymmetry,
                'hip_asymmetry':      hip_asymmetry,
                'joint_velocity':     joint_velocity,
            }
        }
