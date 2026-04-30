# test_pose_analyzer.py

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

import numpy as np
from src.audio_processor import AudioProcessor
from src.sound_detector import SoundDetector
from src.event_filter import AudioEventFilter
from src.frame_extractor import FrameExtractor
from src.face_analyzer import FaceAnalyzer
from src.pose_analyzer import PoseAnalyzer

AUDIO_PATH = "data/extracted/audio.wav"
CHECKPOINT = "models/panns/Cnn14_mAP=0.431.pth"
LABELS_CSV = "models/panns/audioset_labels.csv"
FRAMES_DIR = "data/extracted/frames"

def test_pose_analyzer():
    print("=" * 50)
    print("VERIFICATION TEST")
    print("=" * 50)

    # Phases 3–5: audio pipeline
    print("\n[1/5] Running audio pipeline...")
    processor = AudioProcessor(hop_duration=1.0, normalize=True)
    processed = processor.process(AUDIO_PATH)
    detector = SoundDetector(
        checkpoint_path=CHECKPOINT, labels_csv_path=LABELS_CSV,
        device="cpu", batch_size=16, top_k=3
    )
    detection_result = detector.detect(processed)
    event_filter = AudioEventFilter(
        high_value_threshold=0.40, medium_value_threshold=0.55,
        suppress_sustained_music=True
    )
    filtered_events = event_filter.filter(detection_result)
    print(f"  → {len(filtered_events)} filtered events")

    # Phase 6: frame extraction
    print("\n[2/5] Extracting frames...")
    extractor = FrameExtractor(
        frames_dir=FRAMES_DIR, extraction_fps=1.0,
        pre_window_seconds=1.0, post_window_seconds=2.0
    )
    windows = extractor.extract_for_events(filtered_events)
    print(f"  → {len(windows)} frame windows")

    # Phase 7: face analysis
    print("\n[3/5] Running face analysis...")
    face_analyzer = FaceAnalyzer(max_faces=2, reaction_sensitivity=2.0)
    face_results = face_analyzer.analyze_windows(windows)
    print(f"  → {len(face_results)} face results")

    # Phase 8: pose analysis
    print("\n[4/5] Running pose analysis...")
    pose_analyzer = PoseAnalyzer(reaction_sensitivity=3.0)
    pose_results = pose_analyzer.analyze_windows(windows)

    # Verify
    print("\n[5/5] Verifying results...")
    print("\n--- VERIFICATION RESULTS ---")

    # Check 1: One result per window
    assert len(pose_results) == len(windows), \
        f"FAIL: Expected {len(windows)} results, got {len(pose_results)}"
    print(f"PASS: {len(pose_results)} results (one per event)")

    # Check 2: All scores in [0, 1]
    for r in pose_results:
        assert 0.0 <= r.pose_reaction_score <= 1.0, \
            f"FAIL: Score {r.pose_reaction_score} out of range"
    print("PASS: All pose_reaction_scores in [0.0, 1.0]")

    # Check 3: Non-detected events return 0.5
    no_pose = [r for r in pose_results if not r.pose_detected]
    for r in no_pose:
        assert abs(r.pose_reaction_score - 0.5) < 0.01, \
            f"FAIL: No-pose event should return 0.5, got {r.pose_reaction_score}"
    print(f"PASS: {len(no_pose)} no-pose events return neutral 0.5")

    # Check 4: Detection rates valid
    for r in pose_results:
        assert 0.0 <= r.detection_rate <= 1.0, \
            f"FAIL: detection_rate out of range: {r.detection_rate}"
    print("PASS: All detection rates in [0.0, 1.0]")

    # Display combined face + pose table
    print("\n--- COMBINED FACE + POSE SCORES ---")
    print(
        f"\n{'Event':<22} {'Time':>6} "
        f"{'FaceScore':>10} {'PoseScore':>10} "
        f"{'Δhead':>8} {'Δshoulder':>10} {'Δlean':>7}"
    )
    print("─" * 75)
    for face_r, pose_r in zip(face_results, pose_results):
        mins = int(pose_r.event_timestamp // 60)
        secs = pose_r.event_timestamp % 60
        print(
            f"{pose_r.event_label:<22} "
            f"{mins:02d}:{secs:04.1f} "
            f"{face_r.face_reaction_score:>10.3f} "
            f"{pose_r.pose_reaction_score:>10.3f} "
            f"{pose_r.delta_head:>8.4f} "
            f"{pose_r.delta_shoulder:>10.4f} "
            f"{pose_r.delta_lean:>7.4f}"
        )
    print("─" * 75)

    detected_count = sum(1 for r in pose_results if r.pose_detected)
    reacting_count = sum(
        1 for r in pose_results if r.pose_reaction_score > 0.6
    )

    print(f"  Poses detected: {detected_count}/{len(pose_results)}")
    print(f"  Clear body reactions (>0.6): {reacting_count}/{len(pose_results)}")

if __name__ == "__main__":
    test_pose_analyzer()