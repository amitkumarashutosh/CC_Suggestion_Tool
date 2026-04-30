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
from src.label_generator import LabelGenerator
from src.srt_writer import SRTWriter, seconds_to_srt_timestamp

AUDIO_PATH = "data/extracted/audio.wav"
CHECKPOINT = "models/panns/Cnn14_mAP=0.431.pth"
LABELS_CSV = "models/panns/audioset_labels.csv"
FRAMES_DIR = "data/extracted/frames"
OUTPUT_SRT = "data/output/output.srt"

def test_srt_writer():
    print("=" * 50)
    print("PHASE 12 VERIFICATION TEST")
    print("=" * 50)

    # Test timestamp conversion first
    print("\n[0/6] Testing timestamp conversion...")
    test_cases = [
        (0.0,      "00:00:00,000"),
        (6.0,      "00:00:06,000"),
        (23.5,     "00:00:23,500"),
        (143.75,   "00:02:23,750"),
        (3661.1,   "01:01:01,100"),
        (59.999,   "00:00:59,999"),
        (3600.0,   "01:00:00,000"),
    ]
    for seconds, expected in test_cases:
        result = seconds_to_srt_timestamp(seconds)
        assert result == expected, \
            f"FAIL: {seconds}s → '{result}', expected '{expected}'"
        print(f"  PASS: {seconds:8.3f}s → '{result}'")

    # Phases 3–11
    print("\n[1/6] Running full pipeline (Phases 3–11)...")
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
    generator = LabelGenerator()
    decisions = generator.generate_labels(decisions)
    accepted = [d for d in decisions if d.accepted]
    print(f"  → {len(accepted)} accepted events ready for SRT")

    # Phase 12
    print("\n[2/6] Building SRT blocks...")
    writer = SRTWriter(
        output_path=OUTPUT_SRT,
        min_display_duration=2.0,
        max_display_duration=5.0,
        encoding="utf-8"
    )
    blocks = writer.build_blocks(decisions)
    print(f"  → {len(blocks)} SRT blocks built")

    # Preview before writing
    print("\n[3/6] SRT content preview:")
    print("─" * 40)
    print(writer.preview(blocks))
    print("─" * 40)

    # Write file
    print("\n[4/6] Writing SRT file...")
    output_path = writer.write(blocks)
    print(f"  → Written to: {output_path}")

    # Validate
    print("\n[5/6] Validating SRT file...")
    validation = writer.validate(output_path)

    print("\n--- VERIFICATION RESULTS ---")

    assert validation["valid"], \
        f"FAIL: SRT validation failed:\n" + \
        "\n".join(validation["errors"])
    print(f"PASS: SRT file is valid ({validation['block_count']} blocks)")

    # Check block count matches accepted decisions
    assert validation["block_count"] == len(accepted), \
        (f"FAIL: Expected {len(accepted)} blocks, "
         f"got {validation['block_count']}")
    print(f"PASS: Block count matches accepted decisions ({len(accepted)})")

    # Read and display the actual file content
    print("\n[6/6] Final SRT file contents:")
    print("═" * 40)
    srt_content = open(output_path, encoding="utf-8").read()
    print(srt_content)
    print("═" * 40)

    import os
    file_size = os.path.getsize(output_path)
    print(f"  Output file:  {output_path}")
    print(f"  File size:    {file_size} bytes")
    print(f"  SRT blocks:   {validation['block_count']}")
    print(f"\nThe pipeline is complete end-to-end.")
    print(f"Open {output_path} in VLC alongside your video to verify.")

if __name__ == "__main__":
    test_srt_writer()