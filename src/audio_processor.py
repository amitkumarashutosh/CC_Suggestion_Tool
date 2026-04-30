# =============================================================================
# STAGE 2 (Part 1): Audio Pre-Processing Pipeline
# =============================================================================
# Takes the raw audio.wav produced by ingest.py and outputs:
#   1. Resampled + normalized audio array (32kHz, float32, range [-1, 1])
#   2. A list of overlapping audio windows ready for PANNs inference
#   3. (Optional) Log-Mel spectrogram for visualization and debugging
#
# This module sits between the FFmpeg extraction (Phase 2) and
# the PANNs inference engine (Phase 4).
# =============================================================================

# SECTION: Imports
import numpy as np          # numerical array operations
import librosa              # audio loading, resampling, spectrogram
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import warnings

# Suppress a harmless librosa UserWarning about audioread fallback
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

logger = logging.getLogger("audio_processor")


# SECTION: Data Containers

@dataclass
class AudioWindow:
    """
    A single audio window — a short slice of the full audio array.
    PANNs processes one AudioWindow at a time.

    Fields:
        samples:      numpy array of audio samples (float32, 32kHz)
        start_time:   start of this window in seconds (relative to full audio)
        end_time:     end of this window in seconds
        window_index: position of this window in the sequence (0-based)
    """
    samples: np.ndarray      # shape: (num_samples,) e.g. (64000,) for 2s at 32kHz
    start_time: float        # e.g. 0.0, 1.0, 2.0 ... (hop_size steps)
    end_time: float          # e.g. 2.0, 3.0, 4.0 ...
    window_index: int        # 0, 1, 2, ...


@dataclass
class ProcessedAudio:
    """
    The full output of the audio pre-processing pipeline.
    This is what gets handed to the PANNs inference engine in Phase 4.

    Fields:
        samples:       full normalized audio array (32kHz, float32)
        sample_rate:   always 32000 after pre-processing
        duration:      total duration in seconds
        windows:       list of AudioWindow objects for PANNs inference
        spectrogram:   optional log-mel spectrogram for visualization
        source_path:   path to the original audio.wav file
    """
    samples: np.ndarray
    sample_rate: int
    duration: float
    windows: list            # list[AudioWindow]
    spectrogram: Optional[np.ndarray]  # shape: (n_mels, time_frames) or None
    source_path: str


# SECTION: Audio Processor Class

class AudioProcessor:
    """
    Handles all audio pre-processing operations:
        1. Load WAV file using librosa
        2. Resample to target sample rate (32kHz for PANNs)
        3. Normalize amplitude to [-1, 1] range
        4. Slice into overlapping windows
        5. (Optional) Compute log-mel spectrogram

    Design note: All PANNs-specific parameters (sample rate, window
    duration) are set here with values that match PANNs CNN14's
    training configuration. Do not change these without retraining.
    """

    # These constants match PANNs CNN14's training configuration exactly.
    # Changing them without retraining will degrade classification accuracy.
    PANNS_SAMPLE_RATE = 32000    # PANNs was trained at 32kHz
    PANNS_WINDOW_DURATION = 2.0  # PANNs analyzes 2-second windows

    # Mel spectrogram parameters (match PANNs training config)
    N_MELS = 64          # number of mel frequency bins
    N_FFT = 1024         # FFT window size (in samples)
                         # at 32kHz: 1024/32000 = 32ms per FFT frame
    HOP_LENGTH = 320     # hop between FFT frames (in samples)
                         # at 32kHz: 320/32000 = 10ms hop → 100 frames/sec
    FMIN = 50            # minimum frequency (Hz) — below human speech
    FMAX = 14000         # maximum frequency (Hz) — well above most events

    def __init__(
        self,
        hop_duration: float = 1.0,
        normalize: bool = True,
        compute_spectrogram: bool = False,
        normalization_method: str = "peak"
    ):
        """
        Initialize the audio processor.

        Args:
            hop_duration: Step size between window starts, in seconds.
                          Default 1.0 = 50% overlap with 2s windows.
                          Smaller = more windows = finer time resolution
                          but more computation.
            normalize: If True, normalize amplitude before windowing.
                       Always True in production. False only for debugging
                       to see raw amplitude levels.
            compute_spectrogram: If True, compute log-mel spectrogram
                                 of full audio for visualization.
                                 Adds ~0.5s processing time. Default False
                                 because PANNs computes its own internally.
            normalization_method: "peak" (divide by max) or "rms"
                                  (divide by root-mean-square energy).
                                  "peak" is simpler. "rms" is more robust
                                  to occasional loud spikes in audio.
        """
        self.hop_duration = hop_duration
        self.normalize = normalize
        self.compute_spectrogram = compute_spectrogram
        self.normalization_method = normalization_method

        # Convert durations to sample counts for efficiency
        # (sample counts are what numpy operations actually use)
        self.window_samples = int(
            self.PANNS_WINDOW_DURATION * self.PANNS_SAMPLE_RATE
        )
        # 2.0 seconds × 32000 samples/sec = 64000 samples per window

        self.hop_samples = int(
            self.hop_duration * self.PANNS_SAMPLE_RATE
        )
        # 1.0 second × 32000 samples/sec = 32000 samples per hop

        logger.info(
            f"AudioProcessor initialized: "
            f"target_sr={self.PANNS_SAMPLE_RATE}Hz, "
            f"window={self.PANNS_WINDOW_DURATION}s ({self.window_samples} samples), "
            f"hop={self.hop_duration}s ({self.hop_samples} samples), "
            f"normalize={normalize} ({normalization_method})"
        )


    # ─────────────────────────────────────────────────────────────────────
    # SECTION: Audio Loading
    # ─────────────────────────────────────────────────────────────────────

    def load_audio(self, audio_path: str) -> tuple[np.ndarray, int]:
        """
        Load the WAV file from disk using librosa.

        librosa.load() does three things in one call:
          1. Opens the WAV file
          2. Decodes the PCM samples to float32 in range [-1.0, 1.0]
          3. Resamples to the specified sample rate (sr parameter)

        We load at the original sample rate (sr=None) and resample
        separately — this gives us more control and better error messages.

        Returns:
            samples: numpy float32 array of shape (num_samples,)
            native_sr: original sample rate of the WAV file (e.g. 16000)
        """
        audio_path = str(audio_path)

        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Loading audio: {audio_path}")

        # sr=None tells librosa to NOT resample — load at native sample rate.
        # mono=True forces mono (should already be mono from Phase 2,
        # but this is a safety net).
        # dtype=np.float32 loads samples as 32-bit floats (required by PyTorch).
        samples, native_sr = librosa.load(
            audio_path,
            sr=None,         # preserve native sample rate
            mono=True,       # ensure mono (safety net)
            dtype=np.float32 # float32 required by PyTorch tensors
        )

        duration = len(samples) / native_sr

        logger.info(
            f"Audio loaded: {len(samples):,} samples at {native_sr}Hz "
            f"= {duration:.2f}s"
        )

        return samples, native_sr


    # ─────────────────────────────────────────────────────────────────────
    # SECTION: Resampling
    # ─────────────────────────────────────────────────────────────────────

    def resample(self, samples: np.ndarray, orig_sr: int) -> np.ndarray:
        """
        Resample audio from its native sample rate to PANNS_SAMPLE_RATE (32kHz).

        If audio is already at 32kHz, returns unchanged (no-op).
        If audio is at 16kHz, doubles the number of samples via sinc
        interpolation.

        librosa.resample uses the 'kaiser_best' algorithm by default:
          - Kaiser window FIR filter
          - High quality (minimal aliasing artifacts)
          - Slower than simpler methods but correct

        'Aliasing' is what happens when you resample badly —
        high frequencies fold back into the spectrum as noise.
        The Kaiser filter prevents this.
        """
        if orig_sr == self.PANNS_SAMPLE_RATE:
            logger.info(
                f"Audio already at {self.PANNS_SAMPLE_RATE}Hz. "
                "No resampling needed."
            )
            return samples

        logger.info(
            f"Resampling: {orig_sr}Hz → {self.PANNS_SAMPLE_RATE}Hz"
        )

        # librosa.resample signature:
        # librosa.resample(y, orig_sr=..., target_sr=...)
        # y: input audio array
        # orig_sr: current sample rate
        # target_sr: desired sample rate
        resampled = librosa.resample(
            samples,
            orig_sr=orig_sr,
            target_sr=self.PANNS_SAMPLE_RATE
        )

        logger.info(
            f"Resampling complete: {len(samples):,} samples → "
            f"{len(resampled):,} samples"
        )

        return resampled


    # ─────────────────────────────────────────────────────────────────────
    # SECTION: Normalization
    # ─────────────────────────────────────────────────────────────────────

    def normalize_amplitude(self, samples: np.ndarray) -> np.ndarray:
        """
        Normalize audio amplitude so the loudest sample has magnitude 1.0.

        Two methods available:
          "peak": divide by max absolute value
                  Fast. Simple. Sensitive to outlier spikes.
          "rms":  divide by root-mean-square energy
                  More robust for audio with occasional loud spikes
                  (mic bumps, clipping artifacts in YouTube audio).
                  Better for production use.

        Both methods preserve the shape of the waveform exactly —
        only the scale changes.
        """
        if not self.normalize:
            return samples

        if self.normalization_method == "peak":
            # Find the largest absolute value in the entire audio array
            peak = np.max(np.abs(samples))

            if peak < 1e-8:
                # Audio is essentially silent — avoid division by near-zero
                logger.warning(
                    "Audio appears to be silent (peak amplitude < 1e-8). "
                    "Normalization skipped."
                )
                return samples

            normalized = samples / peak
            logger.info(
                f"Peak normalization: peak was {peak:.4f}, "
                f"now scaled to 1.0"
            )

        elif self.normalization_method == "rms":
            # RMS = Root Mean Square = sqrt(mean(samples²))
            # This is the "average energy" of the signal
            rms = np.sqrt(np.mean(samples ** 2))

            if rms < 1e-8:
                logger.warning(
                    "Audio RMS is near-zero (silent?). "
                    "Normalization skipped."
                )
                return samples

            # Target RMS of 0.1 — a standard level for audio ML models.
            # This ensures the signal is not too quiet (0.0001) or
            # clipping (above 1.0).
            target_rms = 0.1
            normalized = samples * (target_rms / rms)

            # Clip to [-1, 1] to prevent clipping after RMS normalization
            # (RMS normalization can push peaks above 1.0 if the audio
            # has a high peak-to-RMS ratio — common in percussive sounds)
            normalized = np.clip(normalized, -1.0, 1.0)

            logger.info(
                f"RMS normalization: RMS was {rms:.4f}, "
                f"scaled to target {target_rms}"
            )

        else:
            raise ValueError(
                f"Unknown normalization method: {self.normalization_method}. "
                "Choose 'peak' or 'rms'."
            )

        return normalized.astype(np.float32)


    # ─────────────────────────────────────────────────────────────────────
    # SECTION: Windowing
    # ─────────────────────────────────────────────────────────────────────

    def create_windows(self, samples: np.ndarray) -> list:
        """
        Slice the full audio array into overlapping windows.

        Each window is exactly self.window_samples long (64,000 samples
        at 32kHz = 2 seconds). Windows start self.hop_samples apart
        (32,000 samples = 1 second apart at default settings).

        The last window is zero-padded if the audio doesn't divide evenly.
        Zero-padding means appending zeros (silence) to the end of the
        window to bring it to the required length. This is correct —
        silence does not trigger sound event detections.

        Returns a list of AudioWindow objects.

        Visual example with window=2s, hop=1s:

        Audio: [──────────────────────────────] (10 seconds)

        Window 0: [████████]                    0s→2s
        Window 1:     [████████]                1s→3s
        Window 2:         [████████]            2s→4s
        Window 3:             [████████]        3s→5s
        Window 4:                 [████████]    4s→6s
        Window 5:                     [████████]5s→7s
        Window 6:                         [████████]6s→8s
        Window 7:                             [████████]7s→9s
        Window 8:                                 [████████]8s→10s
                                                          (last 1s zero-padded)
        """
        windows = []
        total_samples = len(samples)
        window_index = 0

        # Start position slides forward by hop_samples each iteration
        start_sample = 0

        while start_sample < total_samples:
            end_sample = start_sample + self.window_samples

            if end_sample <= total_samples:
                # Full window — slice directly, no padding needed
                window_samples = samples[start_sample:end_sample].copy()
            else:
                # Partial window at the end of the audio — zero-pad
                # np.zeros creates an array of zeros with the same dtype
                window_samples = np.zeros(
                    self.window_samples,
                    dtype=np.float32
                )
                # Copy whatever samples remain into the start of the window
                remaining = total_samples - start_sample
                window_samples[:remaining] = samples[start_sample:]

            # Convert sample positions to time (seconds)
            start_time = start_sample / self.PANNS_SAMPLE_RATE
            end_time = min(
                end_sample / self.PANNS_SAMPLE_RATE,
                total_samples / self.PANNS_SAMPLE_RATE
            )

            windows.append(AudioWindow(
                samples=window_samples,
                start_time=start_time,
                end_time=end_time,
                window_index=window_index
            ))

            start_sample += self.hop_samples
            window_index += 1

        logger.info(
            f"Windowing complete: {len(windows)} windows "
            f"({self.PANNS_WINDOW_DURATION}s window, "
            f"{self.hop_duration}s hop)"
        )

        return windows


    # ─────────────────────────────────────────────────────────────────────
    # SECTION: Log-Mel Spectrogram
    # ─────────────────────────────────────────────────────────────────────

    def compute_log_mel_spectrogram(
        self,
        samples: np.ndarray
    ) -> np.ndarray:
        """
        Compute the log-mel spectrogram of the audio.

        This is the same computation PANNs does internally to convert
        a raw audio window into the 2D image its CNN processes.
        We expose it here for visualization and debugging.

        Steps:
            1. STFT: convert waveform to frequency domain (complex numbers)
            2. Magnitude: take absolute value (discard phase information)
            3. Mel filterbank: compress to 64 mel frequency bins
            4. Log: apply natural log to compress dynamic range

        Returns:
            spectrogram: numpy array of shape (N_MELS, time_frames)
                         e.g. (64, 640) for 2 seconds at 32kHz
                         with hop_length=100ms → 20 frames/sec
        """
        logger.info("Computing log-mel spectrogram...")

        # Step 1+2: STFT and magnitude in one call via librosa.stft
        # librosa.feature.melspectrogram internally calls librosa.stft
        # n_fft: length of FFT window in samples (1024 = 32ms at 32kHz)
        # hop_length: step between FFT frames in samples (320 = 10ms at 32kHz)
        # n_mels: number of mel frequency bins
        # fmin/fmax: frequency range to analyze
        mel_spectrogram = librosa.feature.melspectrogram(
            y=samples,
            sr=self.PANNS_SAMPLE_RATE,
            n_fft=self.N_FFT,
            hop_length=self.HOP_LENGTH,
            n_mels=self.N_MELS,
            fmin=self.FMIN,
            fmax=self.FMAX
        )
        # mel_spectrogram shape: (N_MELS, time_frames) = (64, ~640)

        # Step 4: Convert to log scale
        # librosa.power_to_db converts power spectrogram to decibels
        # ref=np.max normalizes so the loudest point = 0 dB
        # Everything else is negative dB (quieter than the loudest point)
        log_mel = librosa.power_to_db(mel_spectrogram, ref=np.max)
        # log_mel shape: same as mel_spectrogram — (64, time_frames)

        logger.info(
            f"Spectrogram computed: shape={log_mel.shape}, "
            f"min={log_mel.min():.1f}dB, max={log_mel.max():.1f}dB"
        )

        return log_mel


    # ─────────────────────────────────────────────────────────────────────
    # SECTION: Main Entry Point
    # ─────────────────────────────────────────────────────────────────────

    def process(self, audio_path: str) -> ProcessedAudio:
        """
        Run the full pre-processing pipeline on an audio.wav file.

        Steps:
            1. Load WAV file
            2. Resample to 32kHz
            3. Normalize amplitude
            4. Create overlapping windows
            5. (Optional) Compute log-mel spectrogram

        Returns a ProcessedAudio object containing everything
        PANNs needs for inference.
        """
        logger.info("=" * 60)
        logger.info(f"Starting audio pre-processing: {audio_path}")
        logger.info("=" * 60)

        # Step 1: Load
        samples, native_sr = self.load_audio(audio_path)

        # Step 2: Resample to 32kHz
        samples = self.resample(samples, native_sr)

        # Step 3: Normalize
        samples = self.normalize_amplitude(samples)

        # Step 4: Create windows
        windows = self.create_windows(samples)

        # Step 5: Spectrogram (optional)
        spectrogram = None
        if self.compute_spectrogram:
            spectrogram = self.compute_log_mel_spectrogram(samples)

        duration = len(samples) / self.PANNS_SAMPLE_RATE

        result = ProcessedAudio(
            samples=samples,
            sample_rate=self.PANNS_SAMPLE_RATE,
            duration=duration,
            windows=windows,
            spectrogram=spectrogram,
            source_path=str(audio_path)
        )

        logger.info("=" * 60)
        logger.info("Audio pre-processing complete.")
        logger.info(f"  Samples: {len(samples):,} at {self.PANNS_SAMPLE_RATE}Hz")
        logger.info(f"  Duration: {duration:.2f}s")
        logger.info(f"  Windows: {len(windows)}")
        logger.info(
            f"  Sample range: [{samples.min():.3f}, {samples.max():.3f}]"
        )
        logger.info("=" * 60)

        return result