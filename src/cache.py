# =============================================================================
# Audio Detection Cache
# =============================================================================
# Saves PANNs detection results to disk so re-running the pipeline on the
# same video skips the 8+ second inference step.
#
# Cache key: SHA256 hash of the audio.wav file content
# Cache location: data/extracted/detection_cache.json
# Cache format: JSON (human-readable, inspectable)
# =============================================================================

import json
import hashlib
import logging
from pathlib import Path
from dataclasses import asdict

logger = logging.getLogger("cache")


def compute_audio_hash(audio_path: str) -> str:
    """
    Compute SHA256 hash of audio file content.
    Used as the cache key — same audio = same hash = cache hit.

    SHA256 is collision-resistant: two different audio files will
    not produce the same hash (with overwhelming probability).
    """
    hasher = hashlib.sha256()
    with open(audio_path, "rb") as f:
        # Read in 64KB chunks to handle large files without loading
        # the entire file into RAM
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]  # first 16 chars is enough for our use


def save_detection_cache(
    audio_path: str,
    detection_result,
    cache_dir: str = "data/extracted"
) -> str:
    """
    Save PANNs detection results to a JSON cache file.

    Serializes the DetectionResult's events list to JSON.
    Each event is stored with all its fields.

    Returns the cache file path.
    """
    audio_hash = compute_audio_hash(audio_path)
    cache_path = Path(cache_dir) / f"detection_cache_{audio_hash}.json"

    # Convert events to serializable dicts
    # We exclude 'all_scores' (527 floats per event) to keep cache small
    events_data = []
    for event in detection_result.events:
        events_data.append({
            "label": event.label,
            "audioset_id": event.audioset_id,
            "start_time": event.start_time,
            "end_time": event.end_time,
            "confidence": event.confidence,
            "window_index": event.window_index,
            # all_scores excluded — not needed for downstream phases
        })

    cache_data = {
        "audio_hash": audio_hash,
        "audio_path": str(audio_path),
        "total_windows": detection_result.total_windows,
        "duration": detection_result.duration,
        "device": detection_result.device,
        "event_count": len(events_data),
        "events": events_data
    }

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, indent=2)

    size_kb = cache_path.stat().st_size / 1024
    logger.info(
        f"Detection cache saved: {cache_path} "
        f"({len(events_data)} events, {size_kb:.1f} KB)"
    )
    return str(cache_path)


def load_detection_cache(
    audio_path: str,
    cache_dir: str = "data/extracted"
) -> object:
    """
    Load cached PANNs detection results if available.

    Returns a DetectionResult-like object if cache exists,
    or None if no cache found for this audio file.

    The returned object has the same interface as DetectionResult
    so downstream code doesn't need to know if it came from cache.
    """
    audio_hash = compute_audio_hash(audio_path)
    cache_path = Path(cache_dir) / f"detection_cache_{audio_hash}.json"

    if not cache_path.exists():
        return None

    logger.info(f"Detection cache hit: {cache_path}")

    with open(cache_path, "r", encoding="utf-8") as f:
        cache_data = json.load(f)

    # Reconstruct DetectedEvent objects from cached data
    from src.sound_detector import DetectedEvent, DetectionResult
    import numpy as np

    events = []
    for e in cache_data["events"]:
        events.append(DetectedEvent(
            label=e["label"],
            audioset_id=e["audioset_id"],
            start_time=e["start_time"],
            end_time=e["end_time"],
            confidence=e["confidence"],
            window_index=e["window_index"],
            all_scores=None  # not cached
        ))

    result = DetectionResult(
        events=events,
        total_windows=cache_data["total_windows"],
        duration=cache_data["duration"],
        device=cache_data["device"]
    )

    logger.info(
        f"Cache loaded: {len(events)} events "
        f"(audio_hash={audio_hash})"
    )
    return result