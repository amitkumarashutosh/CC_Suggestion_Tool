# test_label_generator.py

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

from src.audio_processor import AudioProcessor
from src.sound_detector import SoundDetector
from src.event_filter import AudioEventFilter
from src.frame_extractor import FrameExtractor
from src.face_analyzer import FaceAnalyzer
from src.pose_analyzer import PoseAnalyzer
from src.visual_scorer import VisualScorer
from src.decision_engine import CCDecisionEngine
from src.label_generator import LabelGenerator, generate_fallback_label

AUDIO_PATH = "data/extracted/audio.wav"
CHECKPOINT = "models/panns/Cnn14_mAP=0.431.pth"
LABELS_CSV = "models/panns/audioset_labels.csv"
FRAMES_DIR = "data/extracted/frames"

def test_label_generator():
    print("=" * 50)
    print("LABEL GENERATION VERIFICATION TEST")
    print("=" * 50)

    # Phases 3–10
    print("\n[1/5] Running full pipeline (Phases 3–10)...")
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
    extractor = FrameExtractor(
        frames_dir=FRAMES_DIR, extraction_fps=1.0,
        pre_window_seconds=1.0, post_window_seconds=2.0
    )
    windows = extractor.extract_for_events(filtered_events)
    face_analyzer = FaceAnalyzer(max_faces=2, reaction_sensitivity=2.0)
    face_results = face_analyzer.analyze_windows(windows)
    pose_analyzer = PoseAnalyzer(reaction_sensitivity=3.0)
    pose_results = pose_analyzer.analyze_windows(windows)
    scorer = VisualScorer(
        face_weight=0.6, pose_weight=0.4,
        single_signal_discount=0.9, duplicate_time_threshold=1.0
    )
    scored_events = scorer.score(filtered_events, face_results, pose_results)
    engine = CCDecisionEngine(
        audio_weight=0.65, visual_weight=0.35, cc_threshold=0.60,
        high_value_boost=0.05, high_value_boost_min_audio=0.45
    )
    decisions = engine.decide(scored_events)
    print(f"  → {len([d for d in decisions if d.accepted])} accepted decisions")

    # Phase 11
    print("\n[2/5] Generating CC labels...")
    generator = LabelGenerator()
    decisions = generator.generate_labels(decisions)

    # Verify
    print("\n[3/5] Verifying results...")
    print("\n--- VERIFICATION RESULTS ---")

    # Check 1: All decisions have cc_label populated
    for d in decisions:
        assert d.cc_label is not None, \
            f"FAIL: cc_label is None for '{d.label}'"
        assert len(d.cc_label) > 0, \
            f"FAIL: cc_label is empty for '{d.label}'"
    print(f"PASS: All {len(decisions)} decisions have cc_label populated")

    # Check 2: All labels are properly bracketed
    for d in decisions:
        assert d.cc_label.startswith("["), \
            f"FAIL: '{d.cc_label}' does not start with ["
        assert d.cc_label.endswith("]"), \
            f"FAIL: '{d.cc_label}' does not end with ]"
    print("PASS: All CC labels properly bracketed")

    # Check 3: Accepted events have meaningful labels
    accepted = [d for d in decisions if d.accepted]
    for d in accepted:
        # Label should be more than just "[]"
        inner = d.cc_label[1:-1]  # remove brackets
        assert len(inner) >= 2, \
            f"FAIL: CC label too short: '{d.cc_label}'"
    print(f"PASS: All {len(accepted)} accepted events have meaningful labels")

    # Check 4: Test fallback generator directly
    test_cases = [
        ("Ratchet, pawl",    "[Ratchet]"),
        ("Chopping (food)",  "[Chopping]"),
        ("Smash, crash",     "[Smash]"),
        ("Unknown Sound",    "[Unknown Sound]"),
    ]
    print("\n[4/5] Testing fallback label generator...")
    for input_label, expected in test_cases:
        result = generate_fallback_label(input_label)
        assert result == expected, \
            f"FAIL: fallback('{input_label}') = '{result}', expected '{expected}'"
        print(f"  PASS: '{input_label}' → '{result}'")

    # Check 5: Music onset uses context override
    music_decisions = [d for d in decisions if d.label == "Music"]
    for d in music_decisions:
        if d.filter_reason == "music_onset":
            assert d.cc_label == "[Music]", \
                f"FAIL: Music onset should be '[Music]', got '{d.cc_label}'"
    print("\nPASS: Music onset context override works correctly")

    # Display final annotations
    print("\n[5/5] Final CC annotations preview:")
    print("\n" + generator.summarize(decisions))

    print(f"  Labels generated: {len(decisions)}")
    print(f"  Accepted for SRT: {len(accepted)}")

if __name__ == "__main__":
    test_label_generator()