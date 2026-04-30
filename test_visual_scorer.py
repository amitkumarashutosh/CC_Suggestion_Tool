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
from src.visual_scorer import VisualScorer

AUDIO_PATH = "data/extracted/audio.wav"
CHECKPOINT = "models/panns/Cnn14_mAP=0.431.pth"
LABELS_CSV = "models/panns/audioset_labels.csv"
FRAMES_DIR = "data/extracted/frames"

def test_visual_scorer():
    print("=" * 50)
    print("VERIFICATION TEST")
    print("=" * 50)

    # Phases 3–5
    print("\n[1/6] Running audio pipeline...")
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

    # Phase 6
    print("\n[2/6] Extracting frames...")
    extractor = FrameExtractor(
        frames_dir=FRAMES_DIR, extraction_fps=1.0,
        pre_window_seconds=1.0, post_window_seconds=2.0
    )
    windows = extractor.extract_for_events(filtered_events)

    # Phase 7
    print("\n[3/6] Face analysis...")
    face_analyzer = FaceAnalyzer(max_faces=2, reaction_sensitivity=2.0)
    face_results = face_analyzer.analyze_windows(windows)

    # Phase 8
    print("\n[4/6] Pose analysis...")
    pose_analyzer = PoseAnalyzer(reaction_sensitivity=3.0)
    pose_results = pose_analyzer.analyze_windows(windows)

    # Phase 9
    print("\n[5/6] Visual confidence scoring...")
    scorer = VisualScorer(
        face_weight=0.6,
        pose_weight=0.4,
        single_signal_discount=0.9,
        duplicate_time_threshold=1.0
    )
    scored_events = scorer.score(filtered_events, face_results, pose_results)

    # Verify
    print("\n[6/6] Verifying results...")
    print("\n--- VERIFICATION RESULTS ---")

    # Check 1: Correct count
    assert len(scored_events) == len(filtered_events), \
        f"FAIL: Expected {len(filtered_events)} scored events"
    print(f"PASS: {len(scored_events)} scored events")

    # Check 2: All visual_confidence in [0, 1]
    for e in scored_events:
        assert 0.0 <= e.visual_confidence <= 1.0, \
            f"FAIL: visual_confidence {e.visual_confidence} out of range"
    print("PASS: All visual_confidence values in [0.0, 1.0]")

    # Check 3: No-detection events have exactly 0.5 visual confidence
    no_signal = [e for e in scored_events if e.combination_method == "none"]
    for e in no_signal:
        assert abs(e.visual_confidence - 0.5) < 0.001, \
            f"FAIL: No-signal event should be 0.5, got {e.visual_confidence}"
    print(f"PASS: {len(no_signal)} no-signal events have visual_confidence=0.5")

    # Check 4: Bow-wow and Animal deduplication
    bow_wow = [e for e in scored_events if e.label == "Bow-wow"]
    animal  = [e for e in scored_events if e.label == "Animal"]
    if bow_wow and animal:
        assert not bow_wow[0].is_duplicate, \
            "FAIL: Bow-wow should be kept (more specific)"
        assert animal[0].is_duplicate, \
            "FAIL: Animal should be marked duplicate (less specific)"
        print("PASS: Bow-wow kept, Animal marked as duplicate")
    else:
        print("INFO: Bow-wow/Animal deduplication check skipped")

    # Check 5: Dual-signal combination math
    dual_events = [e for e in scored_events if e.combination_method == "dual"]
    for e in dual_events:
        expected = (0.6 * e.face_score) + (0.4 * e.pose_score)
        expected = float(np.clip(expected, 0.0, 1.0))
        assert abs(e.visual_confidence - expected) < 0.001, \
            (f"FAIL: Dual combination wrong for {e.label}: "
             f"expected {expected:.3f}, got {e.visual_confidence:.3f}")
    print(f"PASS: Dual-signal combination math verified ({len(dual_events)} events)")

    # Check 6: ScoredEvent properties work
    first = [e for e in scored_events if not e.is_duplicate][0]
    assert hasattr(first, 'label'), "FAIL: label property missing"
    assert hasattr(first, 'start_time'), "FAIL: start_time property missing"
    assert hasattr(first, 'audio_confidence'), "FAIL: audio_confidence missing"
    print("PASS: ScoredEvent properties accessible")

    # Display summary
    print("\n" + scorer.summarize(scored_events))

    non_dupes = [e for e in scored_events if not e.is_duplicate]
    high_conf = [e for e in non_dupes if e.visual_confidence > 0.6]

    print(f"  Total scored:    {len(scored_events)}")
    print(f"  Unique events:   {len(non_dupes)}")
    print(
        f"  Duplicates:      "
        f"{len(scored_events) - len(non_dupes)}"
    )
    print(f"  High visual>0.6: {len(high_conf)}")

if __name__ == "__main__":
    test_visual_scorer()