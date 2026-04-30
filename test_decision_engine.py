import os
os.environ["GLOG_minloglevel"] = "3"

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

AUDIO_PATH = "data/extracted/audio.wav"
CHECKPOINT = "models/panns/Cnn14_mAP=0.431.pth"
LABELS_CSV = "models/panns/audioset_labels.csv"
FRAMES_DIR = "data/extracted/frames"

def test_decision_engine():
    print("=" * 50)
    print("PHASE 10 VERIFICATION TEST")
    print("=" * 50)

    # Phases 3–5
    print("\n[1/7] Running audio pipeline...")
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
    print("\n[2/7] Extracting frames...")
    extractor = FrameExtractor(
        frames_dir=FRAMES_DIR, extraction_fps=1.0,
        pre_window_seconds=1.0, post_window_seconds=2.0
    )
    windows = extractor.extract_for_events(filtered_events)

    # Phase 7
    print("\n[3/7] Face analysis...")
    face_analyzer = FaceAnalyzer(max_faces=2, reaction_sensitivity=2.0)
    face_results = face_analyzer.analyze_windows(windows)

    # Phase 8
    print("\n[4/7] Pose analysis...")
    pose_analyzer = PoseAnalyzer(reaction_sensitivity=3.0)
    pose_results = pose_analyzer.analyze_windows(windows)

    # Phase 9
    print("\n[5/7] Visual scoring...")
    scorer = VisualScorer(
        face_weight=0.6, pose_weight=0.4,
        single_signal_discount=0.9,
        duplicate_time_threshold=1.0
    )
    scored_events = scorer.score(filtered_events, face_results, pose_results)

    # Phase 10
    print("\n[6/7] Running CC decision engine...")
    engine = CCDecisionEngine(
        audio_weight=0.65,
        visual_weight=0.35,
        cc_threshold=0.60,
        high_value_boost=0.05,
        high_value_boost_min_audio=0.45
    )
    decisions = engine.decide(scored_events)

    # Verify
    print("\n[7/7] Verifying results...")
    print("\n--- VERIFICATION RESULTS ---")

    # Check 1: One decision per scored event
    assert len(decisions) == len(scored_events), \
        f"FAIL: Expected {len(scored_events)} decisions"
    print(f"PASS: {len(decisions)} decisions (one per scored event)")

    # Check 2: Duplicates always rejected
    for d in decisions:
        if d.rejection_reason == "duplicate":
            assert not d.accepted, \
                f"FAIL: Duplicate event should not be accepted: {d.label}"
    print("PASS: All duplicates correctly rejected")

    # Check 3: All cc_scores in [0, 1]
    for d in decisions:
        assert 0.0 <= d.cc_score <= 1.0, \
            f"FAIL: cc_score {d.cc_score} out of range for {d.label}"
    print("PASS: All cc_scores in [0.0, 1.0]")

    # Check 4: Accepted events all have cc_score >= their threshold
    for d in decisions:
        if d.accepted:
            assert d.cc_score >= d.threshold_used - 0.001, \
                (f"FAIL: Accepted event {d.label} has cc_score "
                 f"{d.cc_score:.3f} < threshold {d.threshold_used:.3f}")
    print("PASS: All accepted events have cc_score >= threshold")

    # Check 5: Rejected events all have cc_score < their threshold
    for d in decisions:
        if not d.accepted and d.rejection_reason != "duplicate":
            assert d.cc_score < d.threshold_used + 0.001, \
                (f"FAIL: Rejected event {d.label} has cc_score "
                 f"{d.cc_score:.3f} >= threshold {d.threshold_used:.3f}")
    print("PASS: All rejected events have cc_score < threshold")

    # Check 6: At least some events accepted
    accepted = [d for d in decisions if d.accepted]
    assert len(accepted) > 0, \
        "FAIL: No events accepted — check thresholds"
    print(f"PASS: {len(accepted)} events accepted for CC annotation")

    # Display full decision table
    print("\n" + engine.summarize(decisions))

    # Show what-if analysis
    print("\n" + engine.what_if_threshold(decisions, 0.55))
    print("\n" + engine.what_if_threshold(decisions, 0.65))

    print(f"\n✓ Phase 10 complete.")
    print(f"  Total events:    {len(decisions)}")
    print(f"  Accepted for CC: {len(accepted)}")
    print(
        f"  Rejected:        "
        f"{len(decisions) - len(accepted)}"
    )
    print(f"\nReady for Phase 11: CC label generation.")
    print(f"Accepted events will become SRT annotations:")
    for d in accepted:
        print(f"  [{d.filter_reason}] {d.label} @ {d.timestamp_str}")

if __name__ == "__main__":
    test_decision_engine()