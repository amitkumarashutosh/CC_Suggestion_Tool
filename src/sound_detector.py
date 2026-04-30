# =============================================================================
# STAGE 2 (Part 2): Sound Event Detection with PANNs CNN14
# =============================================================================
# Takes the ProcessedAudio (list of AudioWindows) from audio_processor.py
# and runs PANNs CNN14 inference on each window.
#
# Outputs a list of DetectedEvent objects — each with:
#   - event label (e.g. "Gunshot, gunfire")
#   - start and end timestamp in seconds
#   - audio confidence score (0.0 to 1.0)
#   - full score vector (all 527 class probabilities)
#
# This is the AI core of the pipeline.
# =============================================================================

# SECTION: Imports
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# We need to tell Python where to find the PANNs models.py file.
# sys.path.insert adds a directory to Python's module search path.
# Without this, `import models` would fail because Python doesn't
# know to look in models/panns/.
PANNS_DIR = str(Path(__file__).parent.parent / "models" / "panns")
if PANNS_DIR not in sys.path:
    sys.path.insert(0, PANNS_DIR)

# Now we can import the PANNs model architecture.
# This imports the Cnn14 class definition from models/panns/models.py.
try:
    from models import Cnn14
except ImportError as e:
    raise ImportError(
        f"Could not import PANNs models.py from {PANNS_DIR}.\n"
        f"Ensure models/panns/models.py exists and torchlibrosa is installed.\n"
        f"Run: pip install torchlibrosa\n"
        f"Original error: {e}"
    )

logger = logging.getLogger("sound_detector")


# SECTION: Data Containers

@dataclass
class DetectedEvent:
    """
    A single sound event detected by PANNs in one audio window.

    Fields:
        label:          Human-readable AudioSet class name
                        e.g. "Gunshot, gunfire" or "Laughter"
        audioset_id:    AudioSet class index (0–526)
        start_time:     Start of the audio window in seconds
        end_time:       End of the audio window in seconds
        confidence:     PANNs confidence score for this label (0.0–1.0)
        window_index:   Which window this event came from
        all_scores:     Full 527-class score vector (for debugging)
    """
    label: str
    audioset_id: int
    start_time: float
    end_time: float
    confidence: float
    window_index: int
    all_scores: Optional[np.ndarray] = field(default=None, repr=False)
    # repr=False: don't print all_scores when logging — it's 527 numbers


@dataclass
class DetectionResult:
    """
    Full output of the sound detection stage.

    Fields:
        events:         All detected events (before confidence filtering)
        total_windows:  How many windows were processed
        duration:       Total audio duration processed (seconds)
        device:         'cpu' or 'cuda' (which device PANNs ran on)
    """
    events: list          # list[DetectedEvent]
    total_windows: int
    duration: float
    device: str


# SECTION: AudioSet Label Loader

def load_audioset_labels(labels_csv_path: str) -> dict:
    """
    Load the AudioSet class labels from the CSV file.

    The CSV has columns: index, mid, display_name
      index:        integer 0–526 (class index)
      mid:          AudioSet MID (machine-generated ID, e.g. "/m/07qfr4h")
      display_name: human-readable label (e.g. "Gunshot, gunfire")

    Returns a dict mapping index (int) → display_name (str).

    Example:
        {0: "Speech", 1: "Male speech, man speaking", ..., 526: "Music"}
    """
    if not Path(labels_csv_path).exists():
        raise FileNotFoundError(
            f"AudioSet labels CSV not found: {labels_csv_path}\n"
            "Download it with: wget -O models/panns/audioset_labels.csv "
            "https://raw.githubusercontent.com/qiuqiangkong/"
            "audioset_tagging_cnn/master/metadata/class_labels_indices.csv"
        )

    df = pd.read_csv(labels_csv_path)

    # Build index → label mapping
    # df.iterrows() yields (row_index, row_series) pairs
    label_map = {
        int(row["index"]): str(row["display_name"])
        for _, row in df.iterrows()
    }

    logger.info(f"Loaded {len(label_map)} AudioSet labels from {labels_csv_path}")
    return label_map


# SECTION: PANNs Model Loader

def load_panns_model(
    checkpoint_path: str,
    device: str = "cpu"
) -> tuple:
    """
    Load the PANNs CNN14 model with pretrained weights.

    This function:
      1. Instantiates the CNN14 architecture (empty weights)
      2. Loads the pretrained checkpoint file (.pth file)
      3. Copies the pretrained weights into the model
      4. Sets the model to evaluation mode
      5. Moves the model to the specified device (cpu or cuda)

    Args:
        checkpoint_path: Path to Cnn14_mAP=0.431.pth
        device: 'cpu' or 'cuda'. Use 'cpu' unless you have a GPU.

    Returns:
        (model, device_obj): The loaded model and the torch.device object.

    What is a .pth file?
        A .pth file is a PyTorch checkpoint — a serialized dictionary
        containing the model's learned weight tensors. It's the result
        of training CNN14 on 2 million AudioSet clips over many days
        on multiple GPUs. We load these weights directly — no training
        needed on our side.
    """
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"PANNs checkpoint not found: {checkpoint_path}\n"
            "Download it with the wget command from Phase 1 setup."
        )

    logger.info(f"Loading PANNs CNN14 from: {checkpoint_path}")
    logger.info(f"Device: {device}")

    # Step 1: Determine device
    # torch.device('cpu') creates a CPU device object.
    # torch.device('cuda') creates a GPU device object.
    # We check if CUDA is actually available before trying to use it.
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning(
            "CUDA requested but not available. Falling back to CPU. "
            "Install CUDA and a CUDA-compatible PyTorch build for GPU support."
        )
        device = "cpu"

    device_obj = torch.device(device)

    # Step 2: Instantiate CNN14 architecture with PANNs training parameters.
    # These arguments must match exactly what was used during training.
    # Changing any of these would make the loaded weights incompatible.
    #
    # sample_rate=32000: model was trained on 32kHz audio
    # window_size=1024:  FFT window size for internal spectrogram
    # hop_size=320:      FFT hop size (10ms at 32kHz)
    # mel_bins=64:       number of mel frequency bins
    # fmin=50:           minimum frequency
    # fmax=14000:        maximum frequency
    # classes_num=527:   number of AudioSet classes
    model = Cnn14(
        sample_rate=32000,
        window_size=1024,
        hop_size=320,
        mel_bins=64,
        fmin=50,
        fmax=14000,
        classes_num=527
    )

    # Step 3: Load checkpoint
    # map_location=device_obj tells PyTorch to load tensors onto
    # our target device even if the checkpoint was saved on a different
    # device (e.g. checkpoint was saved on GPU, we're loading on CPU).
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device_obj,
        weights_only=False   # needed for full checkpoint dicts
    )

    # Step 4: Extract and load the model weights.
    # The checkpoint dict has multiple keys:
    #   'model': the actual weight tensors (state_dict)
    #   'iteration': training step when this was saved
    #   'epoch': training epoch
    #   'statistics': validation metrics
    # We only need 'model' (the weights).
    #
    # load_state_dict() copies the weights from the checkpoint into
    # our model's layers. strict=False allows minor mismatches
    # (e.g. if the checkpoint has keys our model doesn't use).
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=False)
        logger.info(
            f"Checkpoint loaded: "
            f"epoch={checkpoint.get('epoch', 'unknown')}, "
            f"iteration={checkpoint.get('iteration', 'unknown')}"
        )
    else:
        # Some checkpoints store weights directly (no outer dict)
        model.load_state_dict(checkpoint, strict=False)
        logger.info("Checkpoint loaded (direct state_dict format).")

    # Step 5: Set to evaluation mode.
    # model.eval() disables two training-specific behaviors:
    #   Dropout: randomly zeroes neurons during training for regularization.
    #            During eval, we want deterministic outputs — no dropout.
    #   BatchNorm: during training, uses batch statistics.
    #              During eval, uses running statistics (more stable).
    # CRITICAL: Always call model.eval() before inference.
    # Forgetting this causes random, non-reproducible outputs.
    model.eval()

    # Step 6: Move model to device
    model = model.to(device_obj)

    # Count parameters to confirm the model loaded correctly
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"PANNs CNN14 loaded: {total_params:,} parameters "
        f"({total_params * 4 / 1024 / 1024:.1f} MB)"
    )

    return model, device_obj


# SECTION: Core Sound Detector

class SoundDetector:
    """
    Runs PANNs CNN14 inference on audio windows and returns
    detected sound events with confidence scores and timestamps.

    Usage:
        detector = SoundDetector(checkpoint_path, labels_csv_path)
        result = detector.detect(processed_audio)
        for event in result.events:
            print(event.label, event.confidence, event.start_time)
    """

    # AudioSet class indices for Speech-related classes.
    # We use these to filter out speech detections — speech has
    # its own subtitle track and does not need CC annotation.
    # Source: AudioSet ontology (https://research.google.com/audioset)
    SPEECH_CLASS_INDICES = {
        0,    # Speech
        1,    # Male speech, man speaking
        2,    # Female speech, woman speaking
        3,    # Child speech, kid speaking
        4,    # Conversation
        5,    # Narration, monologue
        6,    # Babbling
        7,    # Speech synthesizer
        8,    # Shout
        9,    # Bellow
        10,   # Whoop
        11,   # Yell
        12,   # Children shouting
        13,   # Screaming  ← keep this! screaming is a CC event
        # Note: Screaming is debatable. We keep it because "screaming"
        # in a Hindi drama signals narrative events (fear, danger).
        # Remove index 13 from this set if you want to exclude it.
    }
    # Remove screaming from speech filter — it's narratively important
    SPEECH_CLASS_INDICES = SPEECH_CLASS_INDICES - {13}

    def __init__(
        self,
        checkpoint_path: str = "models/panns/Cnn14_mAP=0.431.pth",
        labels_csv_path: str = "models/panns/audioset_labels.csv",
        device: str = "cpu",
        batch_size: int = 16,
        top_k: int = 3,
        filter_speech: bool = True
    ):
        """
        Initialize the sound detector.

        Args:
            checkpoint_path: Path to PANNs CNN14 pretrained weights.
            labels_csv_path: Path to AudioSet class labels CSV.
            device: 'cpu' or 'cuda'.
            batch_size: How many windows to process simultaneously.
                        16 is safe for 8GB RAM. Reduce to 8 if OOM errors.
            top_k: How many top predictions to return per window.
                   top_k=3 returns the 3 highest-scoring classes per window.
                   We return multiple because sounds co-occur: a window
                   with "Music" at 0.85 might also have "Singing" at 0.72.
            filter_speech: If True, skip windows where the top prediction
                           is a Speech class. We don't CC speech events.
        """
        self.batch_size = batch_size
        self.top_k = top_k
        self.filter_speech = filter_speech

        # Load AudioSet labels
        self.label_map = load_audioset_labels(labels_csv_path)

        # Load PANNs model
        self.model, self.device = load_panns_model(checkpoint_path, device)

        logger.info(
            f"SoundDetector ready: batch_size={batch_size}, "
            f"top_k={top_k}, filter_speech={filter_speech}"
        )


    # ─────────────────────────────────────────────────────────────────────
    # SECTION: Batch Inference
    # ─────────────────────────────────────────────────────────────────────

    def _run_batch(self, window_samples_list: list) -> np.ndarray:
        """
        Run PANNs inference on a batch of audio windows.

        Args:
            window_samples_list: List of numpy arrays, each shape (64000,)

        Returns:
            scores: numpy array of shape (batch_size, 527)
                    Each row is the 527 class scores for one window.

        This is an internal method (underscore prefix).
        It handles the numpy → tensor → model → numpy conversion.
        """
        # Stack list of 1D arrays into a 2D batch array
        # np.stack takes [array(64000,), array(64000,), ...] and
        # produces array(batch_size, 64000)
        batch_np = np.stack(window_samples_list, axis=0)
        # batch_np shape: (batch_size, 64000)

        # Convert numpy array to PyTorch tensor
        # torch.FloatTensor creates a float32 tensor
        # .to(self.device) moves it to CPU or GPU
        batch_tensor = torch.FloatTensor(batch_np).to(self.device)
        # batch_tensor shape: (batch_size, 64000)

        # Run inference without computing gradients.
        # torch.no_grad() is a context manager that disables gradient
        # tracking. During inference we don't need gradients (those are
        # only needed during training for backpropagation). Disabling
        # them saves ~50% memory and speeds up inference.
        with torch.no_grad():
            # PANNs forward() returns a dict with multiple outputs.
            # 'clipwise_output': shape (batch_size, 527)
            #    The primary classification output — confidence per class
            #    for the entire clip (window). This is what we use.
            # 'framewise_output': shape (batch_size, time_frames, 527)
            #    Per-frame scores (finer time resolution within the window).
            #    We don't use this in Phase 4, but it exists for Phase 5.
            output_dict = self.model(batch_tensor)
            clipwise_scores = output_dict["clipwise_output"]
            # clipwise_scores shape: (batch_size, 527), values in [0, 1]

        # Move tensor back to CPU and convert to numpy
        # .cpu() moves from GPU to CPU (no-op if already on CPU)
        # .numpy() converts PyTorch tensor to numpy array
        scores_np = clipwise_scores.cpu().numpy()
        # scores_np shape: (batch_size, 527)

        return scores_np


    # ─────────────────────────────────────────────────────────────────────
    # SECTION: Event Extraction from Scores
    # ─────────────────────────────────────────────────────────────────────

    def _extract_events_from_scores(
        self,
        scores: np.ndarray,
        window
    ) -> list:
        """
        Given a 527-element score array for one window, extract the
        top-k predicted sound events as DetectedEvent objects.

        Args:
            scores: numpy array of shape (527,) — one score per AudioSet class
            window: AudioWindow object (has start_time, end_time, window_index)

        Returns:
            List of DetectedEvent objects (up to top_k entries).
            Empty list if top prediction is speech and filter_speech=True.
        """
        # Find the top_k highest scores and their indices
        # np.argsort returns indices that would sort the array in ascending order
        # [::-1] reverses to get descending order (highest first)
        # [:self.top_k] takes only the first top_k indices
        top_k_indices = np.argsort(scores)[::-1][:self.top_k]

        events = []

        for rank, class_idx in enumerate(top_k_indices):
            class_idx = int(class_idx)
            confidence = float(scores[class_idx])
            label = self.label_map.get(class_idx, f"Unknown_{class_idx}")

            # Filter out speech events if requested.
            # We only filter on RANK 0 (the top prediction).
            # If the top prediction is speech, this window is a speech window —
            # skip the entire window, including its secondary predictions.
            # Rationale: if a window is 80% speech and 20% music,
            # it's a speech window. We don't want to CC "Music" in a
            # window that's primarily someone talking.
            if rank == 0 and self.filter_speech:
                if class_idx in self.SPEECH_CLASS_INDICES:
                    logger.debug(
                        f"Window {window.window_index} "
                        f"({window.start_time:.1f}s): "
                        f"Top prediction is speech ({label}, "
                        f"conf={confidence:.3f}). Skipping window."
                    )
                    return []  # Skip this entire window

            event = DetectedEvent(
                label=label,
                audioset_id=class_idx,
                start_time=window.start_time,
                end_time=window.end_time,
                confidence=confidence,
                window_index=window.window_index,
                all_scores=scores  # keep full vector for debugging
            )
            events.append(event)

        return events


    # ─────────────────────────────────────────────────────────────────────
    # SECTION: Main Detection Loop
    # ─────────────────────────────────────────────────────────────────────

    def detect(self, processed_audio) -> DetectionResult:
        """
        Run PANNs inference on all windows in a ProcessedAudio object.

        This is the main method. It:
          1. Groups windows into batches
          2. Runs _run_batch() on each batch
          3. Extracts top-k events from each window's scores
          4. Returns all events in a DetectionResult

        Args:
            processed_audio: ProcessedAudio object from audio_processor.py

        Returns:
            DetectionResult with all detected events.
        """
        windows = processed_audio.windows
        total_windows = len(windows)

        logger.info("=" * 60)
        logger.info(
            f"Starting PANNs inference: {total_windows} windows, "
            f"batch_size={self.batch_size}"
        )
        logger.info("=" * 60)

        all_events = []

        # Process windows in batches
        # range(start, stop, step) generates batch start indices
        # e.g. for 237 windows, batch_size=16:
        # 0, 16, 32, 48, ..., 224, 236 (last batch may be smaller)
        for batch_start in range(0, total_windows, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_windows)
            batch_windows = windows[batch_start:batch_end]

            # Log progress every 5 batches to show the pipeline is running
            if (batch_start // self.batch_size) % 5 == 0:
                progress_pct = (batch_start / total_windows) * 100
                logger.info(
                    f"Processing windows {batch_start}–{batch_end - 1} "
                    f"of {total_windows} ({progress_pct:.0f}%)"
                )

            # Extract just the sample arrays for this batch
            batch_samples = [w.samples for w in batch_windows]

            # Run PANNs on this batch
            # scores shape: (len(batch_windows), 527)
            scores = self._run_batch(batch_samples)

            # Extract events from each window's scores
            for i, window in enumerate(batch_windows):
                window_scores = scores[i]  # shape: (527,)
                events = self._extract_events_from_scores(
                    window_scores, window
                )
                all_events.extend(events)

        logger.info("=" * 60)
        logger.info(
            f"PANNs inference complete: "
            f"{len(all_events)} events detected "
            f"across {total_windows} windows"
        )

        # Log a summary of what was detected
        if all_events:
            # Group events by label and show counts
            from collections import Counter
            label_counts = Counter(e.label for e in all_events)
            logger.info("Top detected sound classes:")
            for label, count in label_counts.most_common(10):
                avg_conf = np.mean([
                    e.confidence for e in all_events if e.label == label
                ])
                logger.info(f"  {label}: {count}x (avg conf: {avg_conf:.3f})")

        logger.info("=" * 60)

        return DetectionResult(
            events=all_events,
            total_windows=total_windows,
            duration=processed_audio.duration,
            device=str(self.device)
        )


    # ─────────────────────────────────────────────────────────────────────
    # SECTION: Convenience — Get Events as DataFrame
    # ─────────────────────────────────────────────────────────────────────

    def events_to_dataframe(self, result: DetectionResult) -> "pd.DataFrame":
        """
        Convert DetectionResult events to a pandas DataFrame.

        Useful for inspection, sorting, and filtering.
        The DataFrame has columns:
            label, audioset_id, start_time, end_time,
            confidence, window_index

        Example output (sorted by confidence desc):
            label              start  end    confidence
            Gunshot, gunfire   12.0   14.0   0.831
            Music              45.0   47.0   0.792
            Laughter           88.0   90.0   0.743
            ...
        """
        if not result.events:
            logger.warning("No events to convert to DataFrame.")
            import pandas as pd
            return pd.DataFrame(columns=[
                "label", "audioset_id", "start_time",
                "end_time", "confidence", "window_index"
            ])

        import pandas as pd
        rows = []
        for event in result.events:
            rows.append({
                "label": event.label,
                "audioset_id": event.audioset_id,
                "start_time": event.start_time,
                "end_time": event.end_time,
                "confidence": round(event.confidence, 4),
                "window_index": event.window_index
            })

        df = pd.DataFrame(rows)
        # Sort by start_time first, then confidence descending
        df = df.sort_values(
            ["start_time", "confidence"],
            ascending=[True, False]
        ).reset_index(drop=True)

        return df