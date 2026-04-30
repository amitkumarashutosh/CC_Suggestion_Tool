import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

import numpy as np
import cv2
from src.audio_processor import AudioProcessor
from src.sound_detector import SoundDetector
from src.event_filter import AudioEventFilter
from src.frame_extractor import FrameExtractor

AUDIO_PATH = "data/extracted/audio.wav"
CHECKPOINT = "models/panns/Cnn14_mAP=0.431.pth"
LABELS_CSV = "models/panns/audioset_labels.csv"
FRAMES_DIR = "data/extracted/frames"

def test_frame_extractor():
    print("=" * 50)
    print("PHASE 6 VERIFICATION TEST")
    print("=" * 50)

    # Phases 3–5: get filtered events
    print("\n[1/4] Running audio pipeline (Phases 3–5)...")
    processor = AudioProcessor(hop_duration=1.0, normalize=True)
    processed = processor.process(AUDIO_PATH)

    detector = SoundDetector(
        checkpoint_path=CHECKPOINT,
        labels_csv_path=LABELS_CSV,
        device="cpu", batch_size=16, top_k=3
    )
    detection_result = detector.detect(processed)

    event_filter = AudioEventFilter(
        high_value_threshold=0.40,
        medium_value_threshold=0.55,
        suppress_sustained_music=True
    )
    filtered_events = event_filter.filter(detection_result)
    print(f"  → {len(filtered_events)} filtered events")

    # Phase 6: extract frames
    print("\n[2/4] Extracting frame windows...")
    extractor = FrameExtractor(
        frames_dir=FRAMES_DIR,
        extraction_fps=1.0,
        pre_window_seconds=1.0,
        post_window_seconds=2.0
    )
    windows = extractor.extract_for_events(filtered_events)

    # Verify
    print("\n[3/4] Verifying results...")
    print("\n--- VERIFICATION RESULTS ---")

    # Check 1: One window per event
    assert len(windows) == len(filtered_events), \
        f"FAIL: Expected {len(filtered_events)} windows, got {len(windows)}"
    print(f"PASS: {len(windows)} frame windows (one per event)")

    # Check 2: All windows have at least some frames
    empty_windows = [w for w in windows if not w.has_frames]
    if empty_windows:
        print(f"WARN: {len(empty_windows)} events have no frames loaded")
    else:
        print("PASS: All events have at least one frame loaded")

    # Check 3: Frames are RGB numpy arrays with correct dtype
    for w in windows:
        for frame in w.frames:
            assert frame.image_rgb.dtype == np.uint8, \
                f"FAIL: dtype should be uint8, got {frame.image_rgb.dtype}"
            assert len(frame.image_rgb.shape) == 3, \
                f"FAIL: shape should be 3D, got {frame.image_rgb.shape}"
            assert frame.image_rgb.shape[2] == 3, \
                f"FAIL: should have 3 channels, got {frame.image_rgb.shape[2]}"
    print("PASS: All frames are (H, W, 3) uint8 numpy arrays")

    # Check 4: Verify BGR→RGB conversion worked
    # Load the same frame with cv2 directly and compare
    if windows and windows[0].frames:
        test_frame = windows[0].frames[0]
        raw_bgr = cv2.imread(test_frame.frame_path)
        raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        assert np.array_equal(test_frame.image_rgb, raw_rgb), \
            "FAIL: BGR→RGB conversion produced wrong result"
        print("PASS: BGR→RGB conversion verified")

    # Check 5: Pre/post frame categorization
    for w in windows:
        e = w.event
        for f in w.pre_frames:
            assert f.timestamp < e.start_time, \
                f"FAIL: Pre-frame at {f.timestamp}s is not before event at {e.start_time}s"
        for f in w.post_frames:
            assert f.timestamp > e.start_time, \
                f"FAIL: Post-frame at {f.timestamp}s is not after event at {e.start_time}s"
    print("PASS: Pre/post frame categorization correct")

    # Check 6: Timestamp calculation
    for w in windows:
        if w.event_frame:
            expected_t = int(w.event.start_time)  # floor to integer (1 FPS)
            actual_t = int(w.event_frame.timestamp)
            assert abs(actual_t - expected_t) <= 1, \
                (f"FAIL: Event frame timestamp {actual_t}s too far from "
                 f"event start {expected_t}s")
    print("PASS: Frame timestamps match event timestamps")

    # Display summary
    print("\n" + extractor.summarize_windows(windows))

    # Display first frame info
    if windows and windows[0].frames:
        first = windows[0].frames[0]
        print(f"\nSample frame details:")
        print(f"  Path:   {first.frame_path}")
        print(f"  Shape:  {first.image_rgb.shape}")
        print(f"  Dtype:  {first.image_rgb.dtype}")
        print(f"  Time:   {first.timestamp:.1f}s")
        print(f"  R[0,0]: {first.image_rgb[0,0,0]}  "
              f"G[0,0]: {first.image_rgb[0,0,1]}  "
              f"B[0,0]: {first.image_rgb[0,0,2]}")

    total_frames = sum(w.frame_count for w in windows)
    print(f"  Frame windows: {len(windows)}")
    print(f"  Total frames loaded: {total_frames}")

if __name__ == "__main__":
    test_frame_extractor()