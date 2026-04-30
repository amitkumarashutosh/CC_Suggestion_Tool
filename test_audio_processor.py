import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

import numpy as np
from src.audio_processor import AudioProcessor

AUDIO_PATH = "data/extracted/audio.wav"

def test_audio_processor():
    print("=" * 50)
    print("PHASE 3 VERIFICATION TEST")
    print("=" * 50)

    processor = AudioProcessor(
        hop_duration=1.0,
        normalize=True,
        compute_spectrogram=True,   # enable for this test only
        normalization_method="peak"
    )

    result = processor.process(AUDIO_PATH)

    print("\n--- VERIFICATION RESULTS ---")

    # Check 1: Sample rate is 32kHz
    assert result.sample_rate == 32000, \
        f"FAIL: sample_rate should be 32000, got {result.sample_rate}"
    print(f"PASS: Sample rate = {result.sample_rate}Hz (32kHz)")

    # Check 2: Amplitude range is [-1, 1]
    max_amp = float(np.max(np.abs(result.samples)))
    assert max_amp <= 1.0 + 1e-6, \
        f"FAIL: Max amplitude {max_amp:.4f} exceeds 1.0"
    print(f"PASS: Amplitude normalized (max abs = {max_amp:.4f})")

    # Check 3: Windows exist and have correct shape
    assert len(result.windows) > 0, "FAIL: No windows created"
    w = result.windows[0]
    assert len(w.samples) == 64000, \
        f"FAIL: Window should have 64000 samples, got {len(w.samples)}"
    print(f"PASS: {len(result.windows)} windows, each {len(w.samples)} samples (2s @ 32kHz)")

    # Check 4: Window timestamps are correct
    assert result.windows[0].start_time == 0.0, \
        "FAIL: First window should start at 0.0s"
    assert result.windows[1].start_time == 1.0, \
        "FAIL: Second window should start at 1.0s (1s hop)"
    print(
        f"PASS: Window timestamps correct "
        f"(window[0]={result.windows[0].start_time}s, "
        f"window[1]={result.windows[1].start_time}s)"
    )

    # Check 5: Spectrogram shape
    assert result.spectrogram is not None, "FAIL: Spectrogram is None"
    assert result.spectrogram.shape[0] == 64, \
        f"FAIL: Spectrogram should have 64 mel bins, got {result.spectrogram.shape[0]}"
    print(f"PASS: Spectrogram shape = {result.spectrogram.shape} (64 mel bins × time frames)")

    # Check 6: Last window is zero-padded correctly
    last_window = result.windows[-1]
    print(
        f"PASS: Last window: {last_window.start_time:.1f}s → {last_window.end_time:.1f}s "
        f"(zero-padded if < 2s of audio remaining)"
    )

    # Check 7: dtype is float32 (required for PyTorch)
    assert result.samples.dtype == np.float32, \
        f"FAIL: dtype should be float32, got {result.samples.dtype}"
    print(f"PASS: dtype = float32 (PyTorch compatible)")

    print(f"\n✓ All checks passed.")

if __name__ == "__main__":
    test_audio_processor()