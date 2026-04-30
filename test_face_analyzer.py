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

AUDIO_PATH = "data/extracted/audio.wav"
CHECKPOINT = "models/panns/Cnn14_mAP=0.431.pth"
LABELS_CSV = "models/panns/audioset_labels.csv"
FRAMES_DIR = "data/extracted/frames"

def test_face_analyzer():
    print("=" * 50)
    print("PHASE 7 VERIFICATION TEST")
    print("=" * 50)

    # Phases 3–5
    print("\n[1/5] Running audio pipeline...")
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

    # Phase 6
    print("\n[2/5] Extracting frame windows...")
    extractor = FrameExtractor(
        frames_dir=FRAMES_DIR,
        extraction_fps=1.0,
        pre_window_seconds=1.0,
        post_window_seconds=2.0
    )
    windows = extractor.extract_for_events(filtered_events)
    print(f"  → {len(windows)} frame windows")

    # Phase 7
    print("\n[3/5] Running face expression analysis...")
    print("      (First run downloads MediaPipe model ~30MB)")
    analyzer = FaceAnalyzer(
        max_faces=2,
        min_detection_confidence=0.5,
        ear_weight=0.4,
        brow_weight=0.35,
        mar_weight=0.25,
        reaction_sensitivity=2.0
    )
    face_results = analyzer.analyze_windows(windows)

    # Verify
    print("\n[4/5] Verifying results...")
    print("\n--- VERIFICATION RESULTS ---")

    # Check 1: One result per window
    assert len(face_results) == len(windows), \
        f"FAIL: Expected {len(windows)} results, got {len(face_results)}"
    print(f"PASS: {len(face_results)} results (one per event)")

    # Check 2: All scores in [0, 1]
    for r in face_results:
        assert 0.0 <= r.face_reaction_score <= 1.0, \
            f"FAIL: Score out of range: {r.face_reaction_score}"
    print("PASS: All face_reaction_scores in [0.0, 1.0]")

    # Check 3: No-face events return neutral score (0.5)
    no_face_results = [r for r in face_results if not r.faces_detected]
    for r in no_face_results:
        assert abs(r.face_reaction_score - 0.5) < 0.01, \
            f"FAIL: No-face event should return 0.5, got {r.face_reaction_score}"
    print(
        f"PASS: {len(no_face_results)} no-face events return neutral 0.5"
    )

    # Check 4: Detection rate in [0, 1]
    for r in face_results:
        assert 0.0 <= r.detection_rate <= 1.0, \
            f"FAIL: detection_rate out of range: {r.detection_rate}"
    print("PASS: All detection rates in [0.0, 1.0]")

    # Display results table
    print("\n[5/5] Results summary:")
    print(f"\n{'Event':<25} {'Time':>6} {'Score':>7} "
          f"{'Detected':>9} {'ΔEAR':>7} {'ΔBrow':>7} {'ΔMAR':>7}")
    print("─" * 70)
    for r in face_results:
        mins = int(r.event_timestamp // 60)
        secs = r.event_timestamp % 60
        print(
            f"{r.event_label:<25} "
            f"{mins:02d}:{secs:04.1f} "
            f"{r.face_reaction_score:>7.3f} "
            f"{'Yes' if r.faces_detected else 'No':>9} "
            f"{r.delta_ear:>+7.3f} "
            f"{r.delta_brow:>+7.3f} "
            f"{r.delta_mar:>+7.3f}"
        )
    print("─" * 70)

    detected_count = sum(1 for r in face_results if r.faces_detected)
    reacting_count = sum(
        1 for r in face_results if r.face_reaction_score > 0.6
    )

    print(f"  Faces detected in: {detected_count}/{len(face_results)} events")
    print(f"  Clear reactions (>0.6): {reacting_count}/{len(face_results)} events")

if __name__ == "__main__":
    test_face_analyzer()