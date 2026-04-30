# =============================================================================
# STAGE 3 (Part 1): Video Frame Extraction at Event Timestamps
# =============================================================================
# For each FilteredEvent from event_filter.py, this module:
#   1. Calculates which frame files correspond to the event timestamp
#   2. Loads those frames from disk using OpenCV
#   3. Converts from BGR (OpenCV default) to RGB (MediaPipe required)
#   4. Returns frames as numpy arrays ready for MediaPipe analysis
#
# This is the bridge between the audio pipeline (Phases 2–5) and
# the visual pipeline (Phases 7–9).
# =============================================================================

# SECTION: Imports
import cv2           # OpenCV: image loading and color conversion
import numpy as np
import logging
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("frame_extractor")


# =============================================================================
# SECTION: Data Containers
# =============================================================================

@dataclass
class FrameData:
    """
    A single video frame loaded from disk and ready for visual analysis.

    Fields:
        frame_path:   Path to the JPEG file on disk
        frame_number: Sequential number (1-based, from filename)
        timestamp:    Time in seconds this frame represents
        image_rgb:    Numpy array, shape (height, width, 3), dtype uint8
                      Color order: RGB (NOT BGR — already converted)
        height:       Frame height in pixels
        width:        Frame width in pixels
    """
    frame_path: str
    frame_number: int
    timestamp: float
    image_rgb: np.ndarray   # shape: (height, width, 3), RGB, uint8
    height: int
    width: int

    def __repr__(self):
        return (
            f"FrameData(frame={self.frame_number}, "
            f"t={self.timestamp:.1f}s, "
            f"shape={self.image_rgb.shape})"
        )


@dataclass
class EventFrameWindow:
    """
    All frames extracted for a single FilteredEvent.
    This is what gets passed to the MediaPipe analyzers in Phase 7+.

    Fields:
        event:         The FilteredEvent this window belongs to
        frames:        List of FrameData objects (chronological order)
        event_frame:   The FrameData closest to the event's start_time
                       (the "anchor" frame — where the sound occurred)
        pre_frames:    Frames before the event timestamp
        post_frames:   Frames after the event timestamp
        missing_count: How many requested frames were not found on disk
                       (can happen at video start/end boundaries)
    """
    event: object            # FilteredEvent (avoid circular import)
    frames: list             # list[FrameData], sorted by timestamp
    event_frame: Optional[object]  # FrameData | None
    pre_frames: list         # list[FrameData] before event timestamp
    post_frames: list        # list[FrameData] after event timestamp
    missing_count: int = 0

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def has_frames(self) -> bool:
        return len(self.frames) > 0

    def __repr__(self):
        return (
            f"EventFrameWindow("
            f"event={self.event.label!r} @ {self.event.start_time:.1f}s, "
            f"frames={self.frame_count}, "
            f"missing={self.missing_count})"
        )


# =============================================================================
# SECTION: Core Frame Extractor
# =============================================================================

class FrameExtractor:
    """
    Loads video frames from the pre-extracted frames directory
    for each audio event timestamp.

    Design decision: We read from pre-extracted JPEG files (Phase 2 output)
    rather than seeking inside the video file directly. This is 20–50×
    faster for sparse frame access patterns (a few frames at specific
    timestamps scattered across a long video).
    """

    def __init__(
        self,
        frames_dir: str = "data/extracted/frames",
        extraction_fps: float = 1.0,
        pre_window_seconds: float = 1.0,
        post_window_seconds: float = 2.0,
    ):
        """
        Initialize the frame extractor.

        Args:
            frames_dir:          Directory containing frame_XXXXXX.jpg files
                                 (output of Phase 2 ingest.py)
            extraction_fps:      FPS at which frames were extracted in Phase 2.
                                 Must match exactly or frame lookup will be wrong.
                                 Default 1.0 = one frame per second.
            pre_window_seconds:  How many seconds before an event to include.
                                 Default 1.0s — captures anticipatory reactions.
            post_window_seconds: How many seconds after an event to include.
                                 Default 2.0s — captures delayed reactions.
                                 Asymmetric: reactions come AFTER the sound.
        """
        self.frames_dir = Path(frames_dir)
        self.extraction_fps = extraction_fps
        self.pre_window_seconds = pre_window_seconds
        self.post_window_seconds = post_window_seconds

        if not self.frames_dir.exists():
            raise FileNotFoundError(
                f"Frames directory not found: {self.frames_dir}\n"
                "Run Phase 2 (ingest.py) first to extract frames."
            )

        # Count available frames for validation
        available = sorted(self.frames_dir.glob("frame_*.jpg"))
        self.total_frames = len(available)
        self.max_timestamp = (self.total_frames - 1) / self.extraction_fps

        if self.total_frames == 0:
            raise FileNotFoundError(
                f"No frame_*.jpg files found in {self.frames_dir}. "
                "Re-run Phase 2 ingestion."
            )

        logger.info(
            f"FrameExtractor initialized: "
            f"{self.total_frames} frames in {self.frames_dir}\n"
            f"  FPS={extraction_fps}, "
            f"window=[-{pre_window_seconds}s, +{post_window_seconds}s], "
            f"max_timestamp={self.max_timestamp:.1f}s"
        )


    # ─────────────────────────────────────────────────────────────────────
    # SECTION: Timestamp → Frame Number Conversion
    # ─────────────────────────────────────────────────────────────────────

    def timestamp_to_frame_number(self, timestamp_seconds: float) -> int:
        """
        Convert a timestamp in seconds to the corresponding frame number.

        Frame numbers are 1-based (FFmpeg starts at frame_000001.jpg).
        At 1 FPS:
          timestamp 0.0s → frame 1
          timestamp 1.0s → frame 2
          timestamp 25.3s → frame 26  (int(25.3 * 1.0) + 1 = 26)
          timestamp 25.9s → frame 26  (int(25.9 * 1.0) + 1 = 26)

        At 5 FPS:
          timestamp 25.3s → frame 127  (int(25.3 * 5.0) + 1 = 127)

        Args:
            timestamp_seconds: Time in seconds (float)

        Returns:
            Frame number (int, 1-based)
        """
        # Clamp timestamp to valid range
        timestamp_seconds = max(0.0, min(timestamp_seconds, self.max_timestamp))

        frame_number = int(timestamp_seconds * self.extraction_fps) + 1
        # +1 because FFmpeg frame numbering starts at 1, not 0

        # Clamp to valid frame range
        frame_number = max(1, min(frame_number, self.total_frames))

        return frame_number


    def frame_number_to_path(self, frame_number: int) -> Path:
        """
        Convert a frame number to its file path.

        frame_number=26 → data/extracted/frames/frame_000026.jpg

        The :06d format pads to 6 digits with leading zeros,
        matching exactly how FFmpeg named the files in Phase 2.
        """
        return self.frames_dir / f"frame_{frame_number:06d}.jpg"


    # ─────────────────────────────────────────────────────────────────────
    # SECTION: Single Frame Loading
    # ─────────────────────────────────────────────────────────────────────

    def load_frame(self, frame_number: int) -> Optional[FrameData]:
        """
        Load a single frame from disk by frame number.

        Steps:
          1. Build the file path from frame_number
          2. Read JPEG with cv2.imread() → BGR numpy array
          3. Convert BGR → RGB with cv2.cvtColor()
          4. Return FrameData with metadata

        Returns None if the frame file doesn't exist.
        This is graceful degradation — not an error — because
        events near the video start/end may request frames
        that don't exist.
        """
        frame_path = self.frame_number_to_path(frame_number)

        if not frame_path.exists():
            logger.debug(f"Frame not found: {frame_path}")
            return None

        # cv2.imread() reads the JPEG and returns a numpy array.
        # Returns None (not raises) if the file is corrupted or unreadable.
        # Shape: (height, width, 3), dtype: uint8, color order: BGR
        bgr_image = cv2.imread(str(frame_path))

        if bgr_image is None:
            logger.warning(
                f"cv2.imread returned None for {frame_path}. "
                "File may be corrupted."
            )
            return None

        # CRITICAL: Convert BGR → RGB.
        # OpenCV loads images in BGR order (Blue channel first).
        # MediaPipe expects RGB order (Red channel first).
        # Without this conversion, face detection accuracy drops
        # significantly because skin tone color profiles are wrong.
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        # rgb_image shape: (height, width, 3), dtype: uint8, color: RGB

        height, width = rgb_image.shape[:2]
        # shape[:2] takes only first 2 dimensions (height, width),
        # ignoring the channel dimension (3)

        # Calculate what timestamp this frame represents
        # Frame 1 → 0.0s, Frame 2 → 1.0s, Frame N → (N-1) / fps
        timestamp = (frame_number - 1) / self.extraction_fps

        return FrameData(
            frame_path=str(frame_path),
            frame_number=frame_number,
            timestamp=timestamp,
            image_rgb=rgb_image,
            height=height,
            width=width
        )


    # ─────────────────────────────────────────────────────────────────────
    # SECTION: Frame Window Loading
    # ─────────────────────────────────────────────────────────────────────

    def get_frame_window(self, event) -> EventFrameWindow:
        """
        Load all frames in the time window around an audio event.

        For event at start_time T:
          Load frames from: T - pre_window_seconds
                        to: T + post_window_seconds

        Args:
            event: FilteredEvent from event_filter.py

        Returns:
            EventFrameWindow with all loaded frames and metadata.
        """
        event_time = event.start_time

        # Calculate the timestamp range for this window
        window_start = max(0.0, event_time - self.pre_window_seconds)
        window_end = min(self.max_timestamp, event_time + self.post_window_seconds)

        # Convert timestamps to frame numbers
        start_frame = self.timestamp_to_frame_number(window_start)
        end_frame = self.timestamp_to_frame_number(window_end)
        event_frame_number = self.timestamp_to_frame_number(event_time)

        logger.debug(
            f"Event '{event.label}' @ {event_time:.1f}s: "
            f"loading frames {start_frame}–{end_frame} "
            f"(window: {window_start:.1f}s → {window_end:.1f}s)"
        )

        # Load all frames in the range
        all_frames = []
        missing_count = 0
        pre_frames = []
        post_frames = []
        event_frame_data = None

        for frame_num in range(start_frame, end_frame + 1):
            frame_data = self.load_frame(frame_num)

            if frame_data is None:
                missing_count += 1
                continue

            all_frames.append(frame_data)

            # Categorize as pre, event, or post frame
            if frame_num < event_frame_number:
                pre_frames.append(frame_data)
            elif frame_num == event_frame_number:
                event_frame_data = frame_data
                # Also add to all_frames (already done above)
            else:
                post_frames.append(frame_data)

        if missing_count > 0:
            logger.warning(
                f"Event '{event.label}' @ {event_time:.1f}s: "
                f"{missing_count} frames missing "
                f"(video boundary or extraction gap)"
            )

        return EventFrameWindow(
            event=event,
            frames=all_frames,
            event_frame=event_frame_data,
            pre_frames=pre_frames,
            post_frames=post_frames,
            missing_count=missing_count
        )


    # ─────────────────────────────────────────────────────────────────────
    # SECTION: Batch Extraction for All Events
    # ─────────────────────────────────────────────────────────────────────

    def extract_for_events(self, filtered_events: list) -> list:
        """
        Extract frame windows for every event in the filtered event list.

        This is the main entry point called by the visual analysis pipeline.

        Args:
            filtered_events: list[FilteredEvent] from event_filter.py

        Returns:
            list[EventFrameWindow], one per event, in the same order
            as filtered_events.
        """
        logger.info("=" * 60)
        logger.info(
            f"Extracting frame windows for "
            f"{len(filtered_events)} events"
        )
        logger.info("=" * 60)

        results = []
        total_frames_loaded = 0

        for i, event in enumerate(filtered_events):
            logger.info(
                f"[{i+1}/{len(filtered_events)}] "
                f"'{event.label}' @ {event.start_time:.1f}s"
            )

            window = self.get_frame_window(event)
            results.append(window)
            total_frames_loaded += window.frame_count

            logger.info(
                f"  → {window.frame_count} frames loaded "
                f"(pre={len(window.pre_frames)}, "
                f"event={1 if window.event_frame else 0}, "
                f"post={len(window.post_frames)})"
            )

            if not window.has_frames:
                logger.warning(
                    f"  ! No frames loaded for event "
                    f"'{event.label}' @ {event.start_time:.1f}s. "
                    f"Visual analysis will be skipped for this event."
                )

        logger.info("=" * 60)
        logger.info(
            f"Frame extraction complete: "
            f"{total_frames_loaded} frames loaded "
            f"across {len(results)} events"
        )
        logger.info("=" * 60)

        return results


    # ─────────────────────────────────────────────────────────────────────
    # SECTION: Utility — Frame Info Summary
    # ─────────────────────────────────────────────────────────────────────

    def summarize_windows(self, windows: list) -> str:
        """
        Generate a human-readable summary of extracted frame windows.
        Useful for debugging and logging.
        """
        if not windows:
            return "No frame windows extracted."

        lines = [
            f"{'─' * 60}",
            f"FRAME WINDOW SUMMARY ({len(windows)} events)",
            f"{'─' * 60}",
        ]

        for w in windows:
            e = w.event
            mins = int(e.start_time // 60)
            secs = e.start_time % 60
            lines.append(
                f"[{mins:02d}:{secs:04.1f}] {e.label:<30s} "
                f"{w.frame_count} frames "
                f"(pre={len(w.pre_frames)}, "
                f"post={len(w.post_frames)}, "
                f"missing={w.missing_count})"
            )
            if w.frames:
                shapes = set(f.image_rgb.shape for f in w.frames)
                for shape in shapes:
                    lines.append(
                        f"         Frame shape: {shape} "
                        f"(H×W×C, RGB, uint8)"
                    )

        lines.append(f"{'─' * 60}")
        return "\n".join(lines)