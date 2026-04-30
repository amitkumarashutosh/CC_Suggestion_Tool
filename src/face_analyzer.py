# src/face_analyzer.py
# =============================================================================
# STAGE 3 (Part 2): Face Expression Analysis with MediaPipe Face Mesh
# Updated for MediaPipe 0.10+ Tasks API
# =============================================================================

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import logging
import urllib.request
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("face_analyzer")


# =============================================================================
# SECTION: MediaPipe Face Mesh Landmark Indices
# These are fixed anatomical constants — do not change.
# =============================================================================

LEFT_EYE_LANDMARKS  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]

LEFT_BROW_TOP    = 105
RIGHT_BROW_TOP   = 334
LEFT_EYE_CENTER  = 159
RIGHT_EYE_CENTER = 386

MOUTH_INNER = [78, 308, 13, 14]


# =============================================================================
# SECTION: Model Download Helper
# =============================================================================

def ensure_face_landmarker_model(
    model_path: str = "models/mediapipe/face_landmarker.task"
) -> str:
    """
    Download the MediaPipe Face Landmarker model if not already present.

    The new Tasks API requires a .task model file downloaded separately.
    This function downloads it once and caches it locally.

    Model source: Google's official MediaPipe model repository.
    Size: ~5MB (much smaller than the old solution models).
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    if os.path.exists(model_path):
        logger.info(f"Face Landmarker model found: {model_path}")
        return model_path

    url = (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    )

    logger.info(f"Downloading Face Landmarker model from Google...")
    logger.info(f"  URL: {url}")
    logger.info(f"  Destination: {model_path}")

    try:
        urllib.request.urlretrieve(url, model_path)
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        logger.info(f"Model downloaded: {size_mb:.1f} MB → {model_path}")
    except Exception as e:
        raise RuntimeError(
            f"Failed to download Face Landmarker model.\n"
            f"URL: {url}\n"
            f"Error: {e}\n"
            f"Ensure you have internet access and write permission to {model_path}"
        )

    return model_path


# =============================================================================
# SECTION: Data Containers
# =============================================================================

@dataclass
class FrameExpressionFeatures:
    """Expression features extracted from a single frame."""
    frame_timestamp: float
    face_detected: bool
    ear: float = 0.0
    brow_raise: float = 0.0
    mar: float = 0.0
    num_faces: int = 0


@dataclass
class FaceReactionResult:
    """Full face analysis result for one EventFrameWindow."""
    event_label: str
    event_timestamp: float
    face_reaction_score: float
    pre_features: list
    post_features: list
    delta_ear: float
    delta_brow: float
    delta_mar: float
    faces_detected: bool
    detection_rate: float

    def __repr__(self):
        return (
            f"FaceReactionResult("
            f"{self.event_label!r} @ {self.event_timestamp:.1f}s, "
            f"score={self.face_reaction_score:.3f}, "
            f"detected={self.faces_detected}, "
            f"Δear={self.delta_ear:+.3f}, "
            f"Δbrow={self.delta_brow:+.3f}, "
            f"Δmar={self.delta_mar:+.3f})"
        )


# =============================================================================
# SECTION: Core Face Analyzer
# =============================================================================

class FaceAnalyzer:
    """
    Runs MediaPipe Face Landmarker (Tasks API, 0.10+) on video frames
    and extracts expression features to detect visual reactions.
    """

    def __init__(
        self,
        max_faces: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        ear_weight: float = 0.4,
        brow_weight: float = 0.35,
        mar_weight: float = 0.25,
        reaction_sensitivity: float = 2.0,
        model_path: str = "models/mediapipe/face_landmarker.task"
    ):
        self.max_faces = max_faces
        self.ear_weight = ear_weight
        self.brow_weight = brow_weight
        self.mar_weight = mar_weight
        self.reaction_sensitivity = reaction_sensitivity

        # Ensure model file is present (downloads if needed)
        model_path = ensure_face_landmarker_model(model_path)

        # Build the Face Landmarker using the Tasks API.
        # FaceLandmarkerOptions replaces the old FaceMesh constructor.
        # running_mode=IMAGE means: treat each frame independently.
        # This is equivalent to static_image_mode=True in the old API.
        base_options = mp_python.BaseOptions(
            model_asset_path=model_path
        )
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=max_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,   # we compute our own features
            output_facial_transformation_matrixes=False
        )

        self.detector = mp_vision.FaceLandmarker.create_from_options(options)

        logger.info(
            f"FaceAnalyzer initialized (Tasks API): "
            f"max_faces={max_faces}, "
            f"weights=(ear={ear_weight}, brow={brow_weight}, "
            f"mar={mar_weight}), "
            f"sensitivity={reaction_sensitivity}"
        )


    def __del__(self):
        """Release MediaPipe resources."""
        if hasattr(self, 'detector'):
            self.detector.close()


    # ─────────────────────────────────────────────────────────────────────
    # SECTION: Landmark Geometry Helpers
    # ─────────────────────────────────────────────────────────────────────

    def _dist(self, landmarks, idx1: int, idx2: int) -> float:
        """Euclidean distance between two landmarks (x, y only)."""
        p1 = landmarks[idx1]
        p2 = landmarks[idx2]
        return float(np.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2))


    def _compute_ear(self, landmarks) -> float:
        """Eye Aspect Ratio averaged across both eyes."""
        # Left eye
        lv1 = self._dist(landmarks, LEFT_EYE_LANDMARKS[1], LEFT_EYE_LANDMARKS[5])
        lv2 = self._dist(landmarks, LEFT_EYE_LANDMARKS[2], LEFT_EYE_LANDMARKS[4])
        lh  = self._dist(landmarks, LEFT_EYE_LANDMARKS[0], LEFT_EYE_LANDMARKS[3])
        left_ear = (lv1 + lv2) / (2.0 * lh + 1e-6)

        # Right eye
        rv1 = self._dist(landmarks, RIGHT_EYE_LANDMARKS[1], RIGHT_EYE_LANDMARKS[5])
        rv2 = self._dist(landmarks, RIGHT_EYE_LANDMARKS[2], RIGHT_EYE_LANDMARKS[4])
        rh  = self._dist(landmarks, RIGHT_EYE_LANDMARKS[0], RIGHT_EYE_LANDMARKS[3])
        right_ear = (rv1 + rv2) / (2.0 * rh + 1e-6)

        return float((left_ear + right_ear) / 2.0)


    def _compute_brow_raise(self, landmarks) -> float:
        """Normalized brow-to-eye distance, both sides averaged."""
        left_brow_y  = landmarks[LEFT_BROW_TOP].y
        left_eye_y   = landmarks[LEFT_EYE_CENTER].y
        right_brow_y = landmarks[RIGHT_BROW_TOP].y
        right_eye_y  = landmarks[RIGHT_EYE_CENTER].y

        # Smaller y = higher on screen → brow raise = eye_y - brow_y
        left_raise  = left_eye_y  - left_brow_y
        right_raise = right_eye_y - right_brow_y
        avg_raise   = (left_raise + right_raise) / 2.0

        # Normalize by inter-eye distance
        inter_eye = self._dist(landmarks, LEFT_EYE_CENTER, RIGHT_EYE_CENTER)
        if inter_eye < 1e-6:
            return 0.0
        return float(avg_raise / inter_eye)


    def _compute_mar(self, landmarks) -> float:
        """Mouth Aspect Ratio using inner lip landmarks."""
        vertical   = self._dist(landmarks, MOUTH_INNER[2], MOUTH_INNER[3])
        horizontal = self._dist(landmarks, MOUTH_INNER[0], MOUTH_INNER[1])
        return float(vertical / (horizontal + 1e-6))


    # ─────────────────────────────────────────────────────────────────────
    # SECTION: Single Frame Analysis
    # ─────────────────────────────────────────────────────────────────────

    def analyze_frame(self, frame_data) -> FrameExpressionFeatures:
        """
        Run face landmark detection on a single frame.

        The Tasks API uses mp.Image instead of raw numpy arrays.
        mp.Image wraps the numpy array with format information.
        image_format=SRGB tells MediaPipe the array is RGB uint8.
        """
        # Wrap numpy array in MediaPipe Image container
        # image_format=mp.ImageFormat.SRGB means: RGB, uint8, standard
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_data.image_rgb
        )

        # Run detection
        # detect() returns a FaceLandmarkerResult object
        result = self.detector.detect(mp_image)

        # result.face_landmarks is:
        #   [] (empty list): no faces detected
        #   [[landmark, ...], ...]: one list of 478 landmarks per face
        #   Note: Tasks API returns 478 landmarks (vs 468 in old API)
        #   The extra 10 are iris landmarks (added in the new model).
        #   Our landmark indices (0–467) remain valid — iris landmarks
        #   are appended at indices 468–477 and we simply ignore them.
        if not result.face_landmarks:
            return FrameExpressionFeatures(
                frame_timestamp=frame_data.timestamp,
                face_detected=False,
                num_faces=0
            )

        num_faces = len(result.face_landmarks)
        # Use the first (primary) face
        primary_landmarks = result.face_landmarks[0]

        ear       = self._compute_ear(primary_landmarks)
        brow_raise = self._compute_brow_raise(primary_landmarks)
        mar       = self._compute_mar(primary_landmarks)

        logger.debug(
            f"Frame {frame_data.frame_number} (t={frame_data.timestamp:.1f}s): "
            f"faces={num_faces}, EAR={ear:.3f}, "
            f"brow={brow_raise:.3f}, MAR={mar:.3f}"
        )

        return FrameExpressionFeatures(
            frame_timestamp=frame_data.timestamp,
            face_detected=True,
            ear=ear,
            brow_raise=brow_raise,
            mar=mar,
            num_faces=num_faces
        )


    # ─────────────────────────────────────────────────────────────────────
    # SECTION: Window Analysis
    # ─────────────────────────────────────────────────────────────────────

    def _average_features(self, feature_list: list) -> tuple:
        """Average EAR, brow, MAR across frames where a face was detected."""
        detected = [f for f in feature_list if f.face_detected]
        if not detected:
            return 0.0, 0.0, 0.0
        return (
            float(np.mean([f.ear        for f in detected])),
            float(np.mean([f.brow_raise for f in detected])),
            float(np.mean([f.mar        for f in detected]))
        )


    def _compute_reaction_score(
        self,
        delta_ear: float,
        delta_brow: float,
        delta_mar: float,
        faces_detected: bool
    ) -> float:
        """Convert feature deltas to a [0,1] reaction score via sigmoid."""
        if not faces_detected:
            return 0.5  # neutral — no visual evidence

        ear_c  = max(0.0, delta_ear)  * self.reaction_sensitivity
        brow_c = max(0.0, delta_brow) * self.reaction_sensitivity
        mar_c  = max(0.0, delta_mar)  * self.reaction_sensitivity

        raw = (
            self.ear_weight  * ear_c +
            self.brow_weight * brow_c +
            self.mar_weight  * mar_c
        )

        # sigmoid: 0 → 0.5 (no change), positive → above 0.5 (reaction)
        return float(1.0 / (1.0 + np.exp(-raw)))


    def analyze_window(self, event_frame_window) -> FaceReactionResult:
        """Analyze all frames in a window and return a FaceReactionResult."""
        event = event_frame_window.event

        if not event_frame_window.has_frames:
            return FaceReactionResult(
                event_label=event.label,
                event_timestamp=event.start_time,
                face_reaction_score=0.5,
                pre_features=[], post_features=[],
                delta_ear=0.0, delta_brow=0.0, delta_mar=0.0,
                faces_detected=False, detection_rate=0.0
            )

        # Analyze pre frames
        pre_features = [
            self.analyze_frame(f)
            for f in event_frame_window.pre_frames
        ]
        # Fallback: use event frame as pre if no pre frames
        if not pre_features and event_frame_window.event_frame:
            pre_features = [self.analyze_frame(event_frame_window.event_frame)]

        # Analyze post frames
        post_features = [
            self.analyze_frame(f)
            for f in event_frame_window.post_frames
        ]

        pre_ear,  pre_brow,  pre_mar  = self._average_features(pre_features)
        post_ear, post_brow, post_mar = self._average_features(post_features)

        delta_ear  = post_ear  - pre_ear
        delta_brow = post_brow - pre_brow
        delta_mar  = post_mar  - pre_mar

        all_features = pre_features + post_features
        faces_detected = any(f.face_detected for f in all_features)
        total_detected = sum(1 for f in all_features if f.face_detected)
        detection_rate = total_detected / len(all_features) if all_features else 0.0

        score = self._compute_reaction_score(
            delta_ear, delta_brow, delta_mar, faces_detected
        )

        result = FaceReactionResult(
            event_label=event.label,
            event_timestamp=event.start_time,
            face_reaction_score=score,
            pre_features=pre_features,
            post_features=post_features,
            delta_ear=delta_ear,
            delta_brow=delta_brow,
            delta_mar=delta_mar,
            faces_detected=faces_detected,
            detection_rate=detection_rate
        )

        logger.info(
            f"'{event.label}' @ {event.start_time:.1f}s: "
            f"face_score={score:.3f} "
            f"(detected={faces_detected}, "
            f"rate={detection_rate:.0%}, "
            f"Δear={delta_ear:+.3f}, "
            f"Δbrow={delta_brow:+.3f}, "
            f"Δmar={delta_mar:+.3f})"
        )

        return result


    def analyze_windows(self, event_frame_windows: list) -> list:
        """Analyze all EventFrameWindows. Returns list[FaceReactionResult]."""
        logger.info("=" * 60)
        logger.info(
            f"Starting face expression analysis: "
            f"{len(event_frame_windows)} event windows"
        )
        logger.info("=" * 60)

        results = [
            self.analyze_window(w)
            for w in event_frame_windows
        ]

        detected  = sum(1 for r in results if r.faces_detected)
        reacting  = sum(1 for r in results if r.face_reaction_score > 0.6)

        logger.info("=" * 60)
        logger.info(f"Face analysis complete: {len(results)} results")
        logger.info(f"  Faces detected in: {detected}/{len(results)} events")
        logger.info(
            f"  Clear reactions (score>0.6): "
            f"{reacting}/{len(results)} events"
        )
        logger.info("=" * 60)

        return results