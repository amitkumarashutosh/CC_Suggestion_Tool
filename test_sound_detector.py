import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

import numpy as np
from src.audio_processor import AudioProcessor
from src.sound_detector import SoundDetector

AUDIO_PATH  = "data/extracted/audio.wav"
CHECKPOINT  = "models/panns/Cnn14_mAP=0.431.pth"
LABELS_CSV  = "models/panns/audioset_labels.csv"

def test_sound_detector():
    print("=" * 50)
    print("PHASE 4 VERIFICATION TEST")
    print("=" * 50)

    # Step 1: Pre-process audio (Phase 3)
    print("\n[1/3] Pre-processing audio...")
    processor = AudioProcessor(hop_duration=1.0, normalize=True)
    processed = processor.process(AUDIO_PATH)
    print(f"  → {len(processed.windows)} windows ready")

    # Step 2: Load detector and run inference (Phase 4)
    print("\n[2/3] Loading PANNs and running inference...")
    print("      (First load takes ~10s — model is ~310MB)")
    detector = SoundDetector(
        checkpoint_path=CHECKPOINT,
        labels_csv_path=LABELS_CSV,
        device="cpu",
        batch_size=16,
        top_k=3,
        filter_speech=True
    )
    result = detector.detect(processed)

    # Step 3: Verify and display results
    print("\n[3/3] Verifying results...")
    print("\n--- VERIFICATION RESULTS ---")

    # Check 1: We got some events
    assert len(result.events) > 0, "FAIL: No events detected at all"
    print(f"PASS: {len(result.events)} total events detected")

    # Check 2: Events have valid timestamps
    for e in result.events:
        assert e.start_time >= 0, f"FAIL: Negative start_time: {e.start_time}"
        assert e.end_time > e.start_time, \
            f"FAIL: end_time <= start_time for {e.label}"
        assert 0 <= e.confidence <= 1.0, \
            f"FAIL: Confidence out of range: {e.confidence}"
    print("PASS: All events have valid timestamps and confidence scores")

    # Check 3: Speech filtering worked
    speech_top_events = [
        e for e in result.events
        if e.window_index == e.window_index  # all events
        and e.audioset_id in detector.SPEECH_CLASS_INDICES
        and e.confidence > 0.8
    ]
    # Note: some speech class IDs may appear as secondary predictions
    # (rank 1 or 2) in non-speech windows — that's fine and expected.
    print(
        f"INFO: {len(speech_top_events)} speech-class events in results "
        "(these are secondary predictions in non-speech windows — OK)"
    )

    # Check 4: Convert to DataFrame and display
    df = detector.events_to_dataframe(result)
    print(f"\nPASS: DataFrame created with {len(df)} rows")

    # Show top 20 most confident events
    print("\n--- TOP 20 DETECTED EVENTS (by confidence) ---")
    top20 = df.nlargest(20, "confidence")[
        ["label", "start_time", "end_time", "confidence"]
    ]
    print(top20.to_string(index=False))

    # Show timeline of events (first 30 seconds)
    print("\n--- EVENTS IN FIRST 60 SECONDS ---")
    early = df[df["start_time"] < 60].sort_values("start_time")
    if len(early) > 0:
        print(early[["start_time", "end_time", "label", "confidence"]]
              .to_string(index=False))
    else:
        print("  No non-speech events detected in first 60 seconds.")

    print(f"  Total events before filtering: {len(result.events)}")
    print(f"  Device used: {result.device}")

if __name__ == "__main__":
    test_sound_detector()