from src.ingest import VideoIngestor
import os

VIDEO_PATH = "data/input/sample_hindi.mp4"

def test_ingestion():
    print("=" * 50)
    print("PHASE 2 VERIFICATION TEST")
    print("=" * 50)

    # Verify the input video exists
    if not os.path.exists(VIDEO_PATH):
        print(f"ERROR: Sample video not found at {VIDEO_PATH}")
        print("Please download a sample video in Phase 1 setup.")
        return

    print(f"Input video found: {VIDEO_PATH}")
    print(f"File size: {os.path.getsize(VIDEO_PATH) / 1024 / 1024:.1f} MB")

    # Run ingestion
    ingestor = VideoIngestor(
        output_dir="data/extracted",
        target_sample_rate=16000,
        target_fps=1.0,
        overwrite=True   # force fresh extraction for this test
    )

    result = ingestor.ingest(VIDEO_PATH)

    # Verify outputs
    print("\n--- VERIFICATION RESULTS ---")

    # Check 1: audio.wav exists
    assert os.path.exists(result.audio_path), "FAIL: audio.wav not found"
    audio_size = os.path.getsize(result.audio_path) / 1024 / 1024
    print(f"PASS: audio.wav exists ({audio_size:.2f} MB)")

    # Check 2: frames directory has files
    from pathlib import Path
    frames = list(Path(result.frames_dir).glob("frame_*.jpg"))
    assert len(frames) > 0, "FAIL: No frames extracted"
    print(f"PASS: {len(frames)} frames extracted")

    # Check 3: duration sanity check
    assert result.duration_seconds > 0, "FAIL: Duration is 0"
    print(f"PASS: Duration = {result.duration_seconds:.1f} seconds")

    # Check 4: frame naming convention
    first_frame = sorted(frames)[0]
    assert first_frame.name == "frame_000001.jpg", (
        f"FAIL: First frame should be frame_000001.jpg, got {first_frame.name}"
    )
    print(f"PASS: Frame naming correct (starts at frame_000001.jpg)")

    # Check 5: frame window helper
    window = ingestor.get_frame_window(
        center_timestamp=result.duration_seconds / 2,  # middle of video
        window_seconds=2.0
    )
    assert len(window) > 0, "FAIL: get_frame_window returned no frames"
    print(f"PASS: get_frame_window works ({len(window)} frames in ±2s window)")

    print("\n✓ All checks passed.")

if __name__ == "__main__":
    test_ingestion()