import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

from src.audio_processor import AudioProcessor
from src.sound_detector import SoundDetector
from src.event_filter import AudioEventFilter

AUDIO_PATH = "data/extracted/audio.wav"
CHECKPOINT = "models/panns/Cnn14_mAP=0.431.pth"
LABELS_CSV = "models/panns/audioset_labels.csv"

def test_event_filter():
    print("=" * 50)
    print("VERIFICATION TEST")
    print("=" * 50)

    # Phase 3: pre-process
    print("\n[1/4] Pre-processing audio...")
    processor = AudioProcessor(hop_duration=1.0, normalize=True)
    processed = processor.process(AUDIO_PATH)

    # Phase 4: detect
    print("\n[2/4] Running PANNs inference...")
    detector = SoundDetector(
        checkpoint_path=CHECKPOINT,
        labels_csv_path=LABELS_CSV,
        device="cpu", batch_size=16, top_k=3
    )
    detection_result = detector.detect(processed)
    print(f"  → {len(detection_result.events)} raw events")

    # Phase 5: filter
    print("\n[3/4] Filtering events...")
    event_filter = AudioEventFilter(
        high_value_threshold=0.40,
        medium_value_threshold=0.55,
        music_onset_threshold=0.60,
        music_silence_threshold=0.25,
        merge_gap_seconds=2.0,
        suppress_sustained_music=True,
        min_music_silence_gap=5.0
    )
    filtered = event_filter.filter(detection_result)

    # Verify
    print("\n[4/4] Verifying results...")
    print("\n--- VERIFICATION RESULTS ---")

    # Check 1: Filtering reduced event count significantly
    raw_count = len(detection_result.events)
    filtered_count = len(filtered)
    assert filtered_count < raw_count, \
        f"FAIL: Filtering should reduce events ({raw_count} → {filtered_count})"
    reduction_pct = (1 - filtered_count / raw_count) * 100
    print(
        f"PASS: Event count reduced: "
        f"{raw_count} → {filtered_count} "
        f"({reduction_pct:.0f}% reduction)"
    )

    # Check 2: No suppressed labels in output
    from src.event_filter import SUPPRESS_ALWAYS
    for e in filtered:
        assert e.label not in SUPPRESS_ALWAYS, \
            f"FAIL: Suppressed label found in output: {e.label}"
    print("PASS: No suppressed labels in filtered output")

    # Check 3: All events have valid filter_reason
    valid_reasons = {"high_value", "medium_value", "music_onset", "unknown"}
    for e in filtered:
        assert e.filter_reason in valid_reasons, \
            f"FAIL: Invalid filter_reason: {e.filter_reason}"
    print("PASS: All events have valid filter_reason")

    # Check 4: Timestamps are sorted
    for i in range(len(filtered) - 1):
        assert filtered[i].start_time <= filtered[i+1].start_time, \
            "FAIL: Events not sorted by start_time"
    print("PASS: Events sorted by start_time")

    # Check 5: No overlapping events of same label
    by_label = {}
    for e in filtered:
        if e.label in by_label:
            prev = by_label[e.label]
            assert e.start_time >= prev.end_time - 0.1, \
                f"FAIL: Overlapping {e.label} events after merging"
        by_label[e.label] = e
    print("PASS: No overlapping same-label events (merging worked)")

    # Display results
    print("\n" + event_filter.summarize(filtered))

    print(f"  Raw events:      {raw_count}")
    print(f"  Filtered events: {filtered_count}")

if __name__ == "__main__":
    test_event_filter()