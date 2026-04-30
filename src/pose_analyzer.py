# src/pose_analyzer.py
# =============================================================================
# STAGE 3 (Part 3): Body Pose and Reaction Analysis with MediaPipe Pose
# =============================================================================
# For each EventFrameWindow from frame_extractor.py, this module:
#   1. Runs MediaPipe Pose Landmarker on every frame in the window
#   2. Extracts key body landmarks (nose, shoulders, hips)
#   3. Computes three reaction signals:
#        - Head movement (nose displacement pre vs post)
#        - Shoulder raise (shoulders moving upward)
#        - Torso lean change (lateral postural shift)
#   4. Compares pre-event vs post-event body position (baseline delta)
#   5. Returns a pose_reaction_score (0.0–1.0) per event window
# =============================================================================

# SECTION: Imports
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import logging
import urllib.request
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("pose_analyzer")


# =============================================================================
# SECTION: MediaPipe Pose Landmark Indices
# These are fixed constants from the MediaPipe Pose topology.
# Source: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
# =============================================================================

# Head / face
NOSE       = 0
LEFT_EAR   = 7
RIGHT_EAR  = 8

# Shoulders — primary reaction indicators
LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12

# Elbows
LEFT_ELBOW  = 13
RIGHT_ELBOW = 14

# Wrists
LEFT_WRIST  = 15
RIGHT_WRIST = 16

# Hips — body anchor point for lean calculation
LEFT_HIP  = 23
RIGHT_HIP = 24

# Minimum visibility to trust a landmark
MIN_VISIBILITY = 0.5


# =============================================================================
# SECTION: Model Download Helper
# =============================================================================

def ensure_pose_landmarker_model(
    model_path: str = "models/mediapipe/pose_landmarker.task"
) -> str:
    """
    Download the MediaPipe Pose Landmarker model if not already present.

    We use the 'lite' variant — fastest, smallest (~6MB), sufficient
    for reaction detection in video frames. The 'full' (~25MB) and
    'heavy' (~95MB) variants offer better accuracy for fine-grained
    pose estimation but are unnecessary for our use case.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    if os.path.exists(model_path):
        logger.info(f"Pose Landmarker model found: {model_path}")
        return model_path

    url = (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_lite/float16/1/"
        "pose_landmarker_lite.task"
    )

    logger.info(f"Downloading Pose Landmarker model (~6MB)...")
    try:
        urllib.request.urlretrieve(url, model_path)
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        logger.info(f"Model downloaded: {size_mb:.1f} MB → {model_path}")
    except Exception as e:
        raise RuntimeError(
            f"Failed to download Pose Landmarker model.\n"
            f"URL: {url}\nError: {e}\n"
            f"Try manually: curl -L -o {model_path} '{url}'"
        )

    return model_path


# =============================================================================
# SECTION: Data Containers
# =============================================================================

@dataclass
class PoseLandmarkSnapshot:
    """
    Key body landmark positions extracted from a single frame.

    Coordinates are normalized [0, 1] relative to frame dimensions.
    Missing landmarks (low visibility or not detected) are stored as None.

    Fields:
        frame_timestamp:   Time of this frame in seconds
        pose_detected:     True if MediaPipe found a person in this frame
        nose_x, nose_y:    Head position (proxy for head direction)
        left_shoulder_x/y: Left shoulder position
        right_shoulder_x/y: Right shoulder position
        left_hip_x/y:      Left hip position
        right_hip_x/y:     Right hip position
        shoulder_visibility: Average visibility of shoulder landmarks
    """
    frame_timestamp: float
    pose_detected: bool

    # Head
    nose_x: Optional[float] = None
    nose_y: Optional[float] = None

    # Shoulders
    left_shoulder_x:  Optional[float] = None
    left_shoulder_y:  Optional[float] = None
    right_shoulder_x: Optional[float] = None
    right_shoulder_y: Optional[float] = None

    # Hips
    left_hip_x:  Optional[float] = None
    left_hip_y:  Optional[float] = None
    right_hip_x: Optional[float] = None
    right_hip_y: Optional[float] = None

    # Confidence
    shoulder_visibility: float = 0.0

    @property
    def shoulder_center_x(self) -> Optional[float]:
        """Midpoint x between both shoulders. None if either missing."""
        if self.left_shoulder_x is None or self.right_shoulder_x is None:
            return None
        return (self.left_shoulder_x + self.right_shoulder_x) / 2.0

    @property
    def shoulder_center_y(self) -> Optional[float]:
        """Midpoint y between both shoulders."""
        if self.left_shoulder_y is None or self.right_shoulder_y is None:
            return None
        return (self.left_shoulder_y + self.right_shoulder_y) / 2.0

    @property
    def hip_center_x(self) -> Optional[float]:
        """Midpoint x between both hips."""
        if self.left_hip_x is None or self.right_hip_x is None:
            return None
        return (self.left_hip_x + self.right_hip_x) / 2.0

    @property
    def hip_center_y(self) -> Optional[float]:
        """Midpoint y between both hips."""
        if self.left_hip_y is None or self.right_hip_y is None:
            return None
        return (self.left_hip_y + self.right_hip_y) / 2.0

    @property
    def torso_lean(self) -> Optional[float]:
        """
        Lateral lean: horizontal offset of shoulder center from hip center.
        Positive = shoulders shifted right of hips.
        Negative = shoulders shifted left of hips.
        Near zero = upright posture.
        None if hips or shoulders not detected.
        """
        sc_x = self.shoulder_center_x
        hc_x = self.hip_center_x
        if sc_x is None or hc_x is None:
            return None
        return sc_x - hc_x


@dataclass
class PoseReactionResult:
    """
    Full pose analysis result for one EventFrameWindow.

    Fields:
        event_label:        Label of the audio event
        event_timestamp:    Start time of the audio event
        pose_reaction_score: Combined reaction score 0.0–1.0
                             0.5 = no pose detected (neutral)
                             <0.5 = pose detected, no reaction
                             >0.5 = pose detected, clear body reaction
        delta_head:         Head displacement (pre vs post)
        delta_shoulder:     Shoulder raise delta (pre vs post)
        delta_lean:         Torso lean change (pre vs post)
        pose_detected:      True if any pose was detected
        detection_rate:     Fraction of frames with pose detected
    """
    event_label: str
    event_timestamp: float
    pose_reaction_score: float
    pre_snapshots: list       # list[PoseLandmarkSnapshot]
    post_snapshots: list      # list[PoseLandmarkSnapshot]
    delta_head: float
    delta_shoulder: float
    delta_lean: float
    pose_detected: bool
    detection_rate: float

    def __repr__(self):
        return (
            f"PoseReactionResult("
            f"{self.event_label!r} @ {self.event_timestamp:.1f}s, "
            f"score={self.pose_reaction_score:.3f}, "
            f"detected={self.pose_detected}, "
            f"Δhead={self.delta_head:.3f}, "
            f"Δshoulder={self.delta_shoulder:.3f}, "
            f"Δlean={self.delta_lean:.3f})"
        )


# =============================================================================
# SECTION: Core Pose Analyzer
# =============================================================================

class PoseAnalyzer:
    """
    Runs MediaPipe Pose Landmarker on video frames and computes
    body reaction scores for audio events.
    """

    def __init__(
        self,
        model_path: str = "models/mediapipe/pose_landmarker.task",
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        head_weight: float = 0.45,
        shoulder_weight: float = 0.35,
        lean_weight: float = 0.20,
        reaction_sensitivity: float = 3.0,
    ):
        """
        Initialize the pose analyzer.

        Args:
            model_path:               Path to pose_landmarker.task
            min_detection_confidence: Minimum confidence for body detection.
            min_tracking_confidence:  Minimum tracking confidence.
            head_weight:              Weight of head movement in score.
                                      Head turns are the most common and
                                      reliable reaction to sudden sounds.
            shoulder_weight:          Weight of shoulder raise in score.
                                      Classic startle/defensive response.
            lean_weight:              Weight of torso lean change.
                                      Less reliable, given lower weight.
            reaction_sensitivity:     Scaling factor for delta scores.
                                      Higher than face analyzer (3.0 vs 2.0)
                                      because body movements in normalized
                                      coordinates are smaller numbers than
                                      face landmark movements.
        """
        self.head_weight = head_weight
        self.shoulder_weight = shoulder_weight
        self.lean_weight = lean_weight
        self.reaction_sensitivity = reaction_sensitivity

        # Ensure model file present
        model_path = ensure_pose_landmarker_model(model_path)

        # Build PoseLandmarker using Tasks API
        # running_mode=IMAGE: each frame analyzed independently
        # (same reasoning as face analyzer — sparse, non-sequential frames)
        base_options = mp_python.BaseOptions(
            model_asset_path=model_path
        )
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,   # detect at most 1 person per frame
                           # reaction detection is always about one
                           # primary subject, not multiple people
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False  # we don't need pixel masks
        )

        self.detector = mp_vision.PoseLandmarker.create_from_options(options)

        logger.info(
            f"PoseAnalyzer initialized: "
            f"weights=(head={head_weight}, shoulder={shoulder_weight}, "
            f"lean={lean_weight}), "
            f"sensitivity={reaction_sensitivity}"
        )


    def __del__(self):
        """Release MediaPipe resources."""
        if hasattr(self, 'detector'):
            self.detector.close()


    # ─────────────────────────────────────────────────────────────────────
    # SECTION: Landmark Extraction Helper
    # ─────────────────────────────────────────────────────────────────────

    def _get_landmark(
        self,
        landmarks: list,
        index: int,
        min_vis: float = MIN_VISIBILITY
    ) -> Optional[tuple]:
        """
        Safely extract a landmark's (x, y) if visibility is sufficient.

        Args:
            landmarks: list of NormalizedLandmark objects from MediaPipe
            index: landmark index (e.g. NOSE=0, LEFT_SHOULDER=11)
            min_vis: minimum visibility threshold

        Returns:
            (x, y) tuple if landmark is visible, None otherwise.

        Why safe extraction matters:
            If a landmark is off-frame or occluded, its x/y coordinates
            are extrapolated by MediaPipe (it guesses where it would be).
            These extrapolated coordinates are unreliable. The visibility
            score tells us when to trust vs ignore a landmark.
        """
        if index >= len(landmarks):
            return None
        lm = landmarks[index]
        if lm.visibility < min_vis:
            return None
        return (lm.x, lm.y)


    # ─────────────────────────────────────────────────────────────────────
    # SECTION: Single Frame Analysis
    # ─────────────────────────────────────────────────────────────────────

    def analyze_frame(self, frame_data) -> PoseLandmarkSnapshot:
        """
        Run MediaPipe Pose on a single frame and extract key landmarks.

        Returns a PoseLandmarkSnapshot with body positions.
        If no person detected, returns a snapshot with pose_detected=False
        and all landmark positions as None.
        """
        # Wrap frame in MediaPipe Image container (same as face analyzer)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_data.image_rgb
        )

        # Run pose detection
        result = self.detector.detect(mp_image)

        # result.pose_landmarks is:
        #   [] (empty list): no pose detected
        #   [[landmark, ...], ...]: one list of 33 landmarks per person
        if not result.pose_landmarks:
            return PoseLandmarkSnapshot(
                frame_timestamp=frame_data.timestamp,
                pose_detected=False
            )

        # Use first detected person's landmarks
        landmarks = result.pose_landmarks[0]
        # landmarks is a list of 33 NormalizedLandmark objects

        # Extract each key landmark with visibility check
        nose_pos         = self._get_landmark(landmarks, NOSE)
        left_sh_pos      = self._get_landmark(landmarks, LEFT_SHOULDER)
        right_sh_pos     = self._get_landmark(landmarks, RIGHT_SHOULDER)
        left_hip_pos     = self._get_landmark(landmarks, LEFT_HIP)
        right_hip_pos    = self._get_landmark(landmarks, RIGHT_HIP)

        # Compute average shoulder visibility for quality assessment
        left_sh_vis  = landmarks[LEFT_SHOULDER].visibility
        right_sh_vis = landmarks[RIGHT_SHOULDER].visibility
        shoulder_vis = (left_sh_vis + right_sh_vis) / 2.0

        snapshot = PoseLandmarkSnapshot(
            frame_timestamp=frame_data.timestamp,
            pose_detected=True,

            # Unpack tuples into x/y fields (or None if not visible)
            nose_x = nose_pos[0]       if nose_pos       else None,
            nose_y = nose_pos[1]       if nose_pos       else None,

            left_shoulder_x  = left_sh_pos[0]   if left_sh_pos   else None,
            left_shoulder_y  = left_sh_pos[1]   if left_sh_pos   else None,
            right_shoulder_x = right_sh_pos[0]  if right_sh_pos  else None,
            right_shoulder_y = right_sh_pos[1]  if right_sh_pos  else None,

            left_hip_x  = left_hip_pos[0]  if left_hip_pos  else None,
            left_hip_y  = left_hip_pos[1]  if left_hip_pos  else None,
            right_hip_x = right_hip_pos[0] if right_hip_pos else None,
            right_hip_y = right_hip_pos[1] if right_hip_pos else None,

            shoulder_visibility = float(shoulder_vis)
        )

        logger.debug(
            f"Frame {frame_data.frame_number} "
            f"(t={frame_data.timestamp:.1f}s): "
            f"pose detected, "
            f"nose=({snapshot.nose_x:.3f}, {snapshot.nose_y:.3f}) "
            f"if nose detected else 'not visible', "
            f"sh_vis={shoulder_vis:.2f}"
        )

        return snapshot


    # ─────────────────────────────────────────────────────────────────────
    # SECTION: Reaction Signal Computation
    # ─────────────────────────────────────────────────────────────────────

    def _average_position(
        self,
        snapshots: list,
        attr_x: str,
        attr_y: str
    ) -> Optional[tuple]:
        """
        Average the (x, y) position of a landmark across multiple snapshots.
        Only includes frames where the landmark was visible (not None).

        Args:
            snapshots: list[PoseLandmarkSnapshot]
            attr_x: attribute name for x coordinate (e.g. "nose_x")
            attr_y: attribute name for y coordinate (e.g. "nose_y")

        Returns:
            (avg_x, avg_y) or None if no valid snapshots.
        """
        valid = [
            (getattr(s, attr_x), getattr(s, attr_y))
            for s in snapshots
            if s.pose_detected
            and getattr(s, attr_x) is not None
            and getattr(s, attr_y) is not None
        ]
        if not valid:
            return None
        avg_x = np.mean([v[0] for v in valid])
        avg_y = np.mean([v[1] for v in valid])
        return (float(avg_x), float(avg_y))


    def _compute_head_displacement(
        self,
        pre_snapshots: list,
        post_snapshots: list
    ) -> float:
        """
        Compute the Euclidean displacement of the nose position
        between pre and post frames.

        Head turning toward a sound source moves the nose laterally.
        Ducking from a loud sound moves it vertically.
        Both are captured by Euclidean distance.

        Returns 0.0 if nose not detected in pre or post frames.
        """
        pre_nose  = self._average_position(pre_snapshots,  "nose_x", "nose_y")
        post_nose = self._average_position(post_snapshots, "nose_x", "nose_y")

        if pre_nose is None or post_nose is None:
            return 0.0

        displacement = np.sqrt(
            (post_nose[0] - pre_nose[0]) ** 2 +
            (post_nose[1] - pre_nose[1]) ** 2
        )
        return float(displacement)


    def _compute_shoulder_raise(
        self,
        pre_snapshots: list,
        post_snapshots: list
    ) -> float:
        """
        Compute the vertical change in shoulder center position.

        In normalized coordinates, y increases downward.
        So shoulders RISING = shoulder_y DECREASING.
        delta_shoulder = pre_shoulder_y - post_shoulder_y
        Positive delta = shoulders moved upward (raised) = startle reaction.
        Negative delta = shoulders dropped = relaxation.

        We only use positive delta (shoulders raising) as a reaction signal.
        """
        pre_sh  = self._average_position(
            pre_snapshots,  "left_shoulder_y", "right_shoulder_y"
        )
        post_sh = self._average_position(
            post_snapshots, "left_shoulder_y", "right_shoulder_y"
        )

        # Fallback: use shoulder_center_y property
        def avg_shoulder_y(snapshots):
            vals = [
                s.shoulder_center_y for s in snapshots
                if s.pose_detected and s.shoulder_center_y is not None
            ]
            return float(np.mean(vals)) if vals else None

        pre_y  = avg_shoulder_y(pre_snapshots)
        post_y = avg_shoulder_y(post_snapshots)

        if pre_y is None or post_y is None:
            return 0.0

        # pre_y - post_y: positive = shoulders moved up (raised)
        return float(pre_y - post_y)


    def _compute_lean_change(
        self,
        pre_snapshots: list,
        post_snapshots: list
    ) -> float:
        """
        Compute the change in torso lateral lean between pre and post frames.

        Lean = shoulder_center_x - hip_center_x
        Change in lean = |post_lean - pre_lean|
        We use absolute value: any lean direction (left or right) is a reaction.

        Returns 0.0 if hips or shoulders not detected.
        """
        def avg_lean(snapshots):
            vals = [
                s.torso_lean for s in snapshots
                if s.pose_detected and s.torso_lean is not None
            ]
            return float(np.mean(vals)) if vals else None

        pre_lean  = avg_lean(pre_snapshots)
        post_lean = avg_lean(post_snapshots)

        if pre_lean is None or post_lean is None:
            return 0.0

        # Absolute change in lean angle
        return float(abs(post_lean - pre_lean))


    # ─────────────────────────────────────────────────────────────────────
    # SECTION: Reaction Score
    # ─────────────────────────────────────────────────────────────────────

    def _compute_reaction_score(
        self,
        delta_head: float,
        delta_shoulder: float,
        delta_lean: float,
        pose_detected: bool
    ) -> float:
        """
        Convert body movement deltas into a [0, 1] reaction score.

        Same sigmoid approach as face analyzer:
          No pose detected → 0.5 (neutral, no evidence)
          No movement      → sigmoid(0) = 0.5
          Movement present → sigmoid(positive) > 0.5

        Body movements in normalized coordinates are smaller numbers
        than face landmark movements (body covers more of the frame,
        so relative movements are smaller fractions).
        Higher reaction_sensitivity (3.0) compensates for this.
        """
        if not pose_detected:
            return 0.5

        # Only count meaningful movements (floor negatives at 0)
        # delta_shoulder can be negative (shoulders dropped) — not a reaction
        head_c     = max(0.0, delta_head)     * self.reaction_sensitivity
        shoulder_c = max(0.0, delta_shoulder) * self.reaction_sensitivity
        lean_c     = max(0.0, delta_lean)     * self.reaction_sensitivity
        # Note: lean is already absolute value, so always >= 0

        raw_score = (
            self.head_weight     * head_c +
            self.shoulder_weight * shoulder_c +
            self.lean_weight     * lean_c
        )

        # Sigmoid: raw=0 → 0.5, raw=1 → 0.73, raw=2 → 0.88
        return float(1.0 / (1.0 + np.exp(-raw_score)))


    # ─────────────────────────────────────────────────────────────────────
    # SECTION: Window Analysis
    # ─────────────────────────────────────────────────────────────────────

    def analyze_window(self, event_frame_window) -> PoseReactionResult:
        """
        Analyze all frames in an EventFrameWindow and compute
        the pose_reaction_score for that audio event.
        """
        event = event_frame_window.event

        if not event_frame_window.has_frames:
            logger.warning(
                f"No frames for '{event.label}' @ {event.start_time:.1f}s. "
                "Returning neutral score."
            )
            return PoseReactionResult(
                event_label=event.label,
                event_timestamp=event.start_time,
                pose_reaction_score=0.5,
                pre_snapshots=[], post_snapshots=[],
                delta_head=0.0, delta_shoulder=0.0, delta_lean=0.0,
                pose_detected=False, detection_rate=0.0
            )

        # Analyze pre frames
        pre_snapshots = [
            self.analyze_frame(f)
            for f in event_frame_window.pre_frames
        ]

        # Fallback: use event frame as pre baseline
        if not pre_snapshots and event_frame_window.event_frame:
            pre_snapshots = [
                self.analyze_frame(event_frame_window.event_frame)
            ]

        # Analyze post frames
        post_snapshots = [
            self.analyze_frame(f)
            for f in event_frame_window.post_frames
        ]

        # Compute the three reaction signals
        delta_head     = self._compute_head_displacement(
            pre_snapshots, post_snapshots
        )
        delta_shoulder = self._compute_shoulder_raise(
            pre_snapshots, post_snapshots
        )
        delta_lean     = self._compute_lean_change(
            pre_snapshots, post_snapshots
        )

        # Determine detection statistics
        all_snapshots = pre_snapshots + post_snapshots
        pose_detected = any(s.pose_detected for s in all_snapshots)
        total_detected = sum(1 for s in all_snapshots if s.pose_detected)
        detection_rate = (
            total_detected / len(all_snapshots) if all_snapshots else 0.0
        )

        # Compute combined reaction score
        score = self._compute_reaction_score(
            delta_head, delta_shoulder, delta_lean, pose_detected
        )

        result = PoseReactionResult(
            event_label=event.label,
            event_timestamp=event.start_time,
            pose_reaction_score=score,
            pre_snapshots=pre_snapshots,
            post_snapshots=post_snapshots,
            delta_head=delta_head,
            delta_shoulder=delta_shoulder,
            delta_lean=delta_lean,
            pose_detected=pose_detected,
            detection_rate=detection_rate
        )

        logger.info(
            f"'{event.label}' @ {event.start_time:.1f}s: "
            f"pose_score={score:.3f} "
            f"(detected={pose_detected}, "
            f"rate={detection_rate:.0%}, "
            f"Δhead={delta_head:.4f}, "
            f"Δshoulder={delta_shoulder:.4f}, "
            f"Δlean={delta_lean:.4f})"
        )

        return result


    # ─────────────────────────────────────────────────────────────────────
    # SECTION: Batch Analysis
    # ─────────────────────────────────────────────────────────────────────

    def analyze_windows(self, event_frame_windows: list) -> list:
        """
        Analyze all EventFrameWindows. Returns list[PoseReactionResult].
        """
        logger.info("=" * 60)
        logger.info(
            f"Starting pose reaction analysis: "
            f"{len(event_frame_windows)} event windows"
        )
        logger.info("=" * 60)

        results = []
        for i, window in enumerate(event_frame_windows):
            logger.info(
                f"[{i+1}/{len(event_frame_windows)}] "
                f"Analyzing '{window.event.label}' "
                f"@ {window.event.start_time:.1f}s"
            )
            result = self.analyze_window(window)
            results.append(result)

        detected = sum(1 for r in results if r.pose_detected)
        reacting = sum(1 for r in results if r.pose_reaction_score > 0.6)

        logger.info("=" * 60)
        logger.info(f"Pose analysis complete: {len(results)} results")
        logger.info(f"  Poses detected in: {detected}/{len(results)} events")
        logger.info(
            f"  Clear reactions (score>0.6): "
            f"{reacting}/{len(results)} events"
        )
        logger.info("=" * 60)

        return results