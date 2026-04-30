# =============================================================================
# STAGE 2 (Part 3): Audio Event Filtering and Confidence Thresholding
# =============================================================================
# Takes the raw DetectionResult from sound_detector.py (636 events) and:
#   1. Applies confidence threshold filtering
#   2. Merges overlapping/adjacent windows with the same label
#   3. Applies CC relevance filtering (suppresses ambient/continuous sounds)
#   4. Detects music onsets (suppresses sustained music, keeps transitions)
#
# Output: a clean list of FilteredEvent objects ready for visual
# confirmation in Phase 6 onwards.
# =============================================================================

import numpy as np
import logging
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional

logger = logging.getLogger("event_filter")


# =============================================================================
# SECTION: CC Relevance Classification
# =============================================================================
# These sets define editorial policy: which AudioSet labels are worth
# captioning. Editors can extend these via config.yaml in Phase 13.
# Label strings must match AudioSet display_name exactly (case-sensitive).

# Events that are ALWAYS worth a CC regardless of context
HIGH_VALUE_CC_LABELS = {
    # Impact / collision
    "Gunshot, gunfire", "Gunshot", "Explosion", "Boom",
    "Glass", "Shatter", "Breaking", "Bang", "Slam", "Crash",
    "Smash, crash", "Thud",

    # Human emotional reactions (narratively important)
    "Laughter", "Giggling", "Chuckling, kn chuckling",
    "Crying, sobbing", "Whimper", "Screaming",
    "Applause", "Clapping",
    "Crowd", "Cheering",

    # Animal sounds (scene-setting)
    "Dog", "Bark", "Bow-wow", "Animal",
    "Cat", "Meow", "Bird", "Chirp", "Crow",

    # Alarms and signals
    "Telephone", "Telephone bell ringing", "Ringtone",
    "Alarm", "Siren", "Doorbell", "Bell",
    "Smoke detector, smoke alarm",

    # Mechanical / vehicle events (distinct, sudden)
    "Engine starting", "Brake", "Skidding",
    "Honk", "Car alarm",

    # Weather events (scene-setting)
    "Thunder", "Lightning",
    "Rain on surface",
}

# Events that are CONTEXT-DEPENDENT — only caption above a higher threshold
MEDIUM_VALUE_CC_LABELS = {
    # Vehicles in motion (often ambient)
    "Vehicle", "Car", "Motorcycle", "Train", "Bus",
    "Aircraft", "Helicopter",

    # Domestic sounds (worth captioning if prominent)
    "Chopping (food)", "Mechanisms", "Ratchet, pawl",
    "Pulleys", "Typing", "Writing",
    "Door", "Knock",

    # Crowd / environment
    "Crowd", "Hubbub, speech noise, speech babble",
    "Background noise",

    # Beeps and electronic sounds
    "Beep, bleep", "Clicking",
}

# Events that are ALWAYS SUPPRESSED (never worth a CC)
# These are either speech (handled by subtitle track) or
# continuous ambient sounds that don't signal narrative events
SUPPRESS_ALWAYS = {
    # Speech — handled by subtitle track
    "Speech", "Male speech, man speaking",
    "Female speech, woman speaking",
    "Child speech, kid speaking",
    "Conversation", "Narration, monologue",
    "Babbling",

    # Continuous ambients — never meaningful as a CC
    "Silence", "White noise", "Static", "Noise",
    "Reverberation", "Echo",
    "Inside, small room", "Inside, large room or hall",
    "Outside, urban or manmade",
    "Outside, rural or natural",

    # Music subgenres — handled by music onset detection separately
    # (suppress individual subgenre labels, keep onset detection)
    "Punk rock", "Grunge", "Reggae", "Hip hop music",
    "Pop music", "Funk", "Rapping", "Singing",
    "Song", "Male singing", "Female singing",
    "Music of Bollywood", "Mantra",
    "Musical instrument",
}

# Music-related labels for onset detection
MUSIC_LABELS = {
    "Music", "Music of Bollywood", "Mantra",
    "Tabla music", "Classical music",
}


# =============================================================================
# SECTION: Data Container
# =============================================================================

@dataclass
class FilteredEvent:
    """
    A single sound event that has passed all filtering criteria
    and is ready for visual confirmation (Phase 6+).

    Fields:
        label:           AudioSet class name (e.g. "Bow-wow")
        cc_label:        Human-readable CC text (e.g. "[Dog Barking]")
                         Set to None here — populated in Phase 11
        start_time:      Event start in seconds
        end_time:        Event end in seconds
        peak_confidence: Highest PANNs confidence across merged windows
        avg_confidence:  Average confidence across merged windows
        window_count:    How many overlapping windows were merged
        filter_reason:   Why this event passed ("high_value", "medium_value",
                         "music_onset")
    """
    label: str
    cc_label: Optional[str]
    start_time: float
    end_time: float
    peak_confidence: float
    avg_confidence: float
    window_count: int
    filter_reason: str

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def __repr__(self):
        return (
            f"FilteredEvent({self.label!r}, "
            f"{self.start_time:.1f}s→{self.end_time:.1f}s, "
            f"conf={self.peak_confidence:.3f}, "
            f"reason={self.filter_reason!r})"
        )


# =============================================================================
# SECTION: Core Filter Class
# =============================================================================

class AudioEventFilter:
    """
    Applies three-stage filtering to raw PANNs detections:
      Stage A: Confidence threshold filtering
      Stage B: Window merging
      Stage C: CC relevance filtering + music onset detection

    All thresholds are configurable at construction time.
    Phase 13 will wire these to config.yaml.
    """

    def __init__(
        self,
        # Stage A: Confidence thresholds
        high_value_threshold: float = 0.40,
        medium_value_threshold: float = 0.55,
        music_onset_threshold: float = 0.60,
        music_silence_threshold: float = 0.25,

        # Stage B: Merging
        merge_gap_seconds: float = 2.0,

        # Stage C: Music suppression
        suppress_sustained_music: bool = True,
        min_music_silence_gap: float = 5.0,

        # Extra suppression control
        extra_suppress_labels: Optional[set] = None,
        extra_high_value_labels: Optional[set] = None,
    ):
        """
        Args:
            high_value_threshold:    Min confidence for HIGH_VALUE events.
                                     Lower because these events are always
                                     worth captioning even at moderate conf.
            medium_value_threshold:  Min confidence for MEDIUM_VALUE events.
                                     Higher because these are context-dependent.
            music_onset_threshold:   Min Music confidence to count as "music
                                     present" during onset detection.
            music_silence_threshold: Max Music confidence to count as "silence"
                                     (no music) before an onset.
            merge_gap_seconds:       Max gap between same-label events to merge.
                                     2.0s means: if two "Bow-wow" events are
                                     within 2s of each other, merge them.
            suppress_sustained_music: If True, suppress Music events that are
                                      not onsets. If False, keep all Music.
            min_music_silence_gap:   How long (seconds) music must be absent
                                     before the next music detection counts
                                     as a new onset. Prevents re-triggering
                                     on brief dips in continuous music.
            extra_suppress_labels:   Additional labels to suppress (from config)
            extra_high_value_labels: Additional high-value labels (from config)
        """
        self.high_value_threshold = high_value_threshold
        self.medium_value_threshold = medium_value_threshold
        self.music_onset_threshold = music_onset_threshold
        self.music_silence_threshold = music_silence_threshold
        self.merge_gap_seconds = merge_gap_seconds
        self.suppress_sustained_music = suppress_sustained_music
        self.min_music_silence_gap = min_music_silence_gap

        # Build effective label sets by merging defaults with extras
        self.suppress_set = SUPPRESS_ALWAYS.copy()
        if extra_suppress_labels:
            self.suppress_set.update(extra_suppress_labels)

        self.high_value_set = HIGH_VALUE_CC_LABELS.copy()
        if extra_high_value_labels:
            self.high_value_set.update(extra_high_value_labels)

        logger.info(
            f"AudioEventFilter initialized:\n"
            f"  high_value_threshold={high_value_threshold}, "
            f"medium_value_threshold={medium_value_threshold}\n"
            f"  merge_gap={merge_gap_seconds}s, "
            f"suppress_sustained_music={suppress_sustained_music}"
        )


    # ─────────────────────────────────────────────────────────────────────
    # STAGE A: Confidence Threshold Filtering
    # ─────────────────────────────────────────────────────────────────────

    def _apply_confidence_threshold(self, events: list) -> list:
        """
        Remove events below the confidence threshold for their category.

        HIGH_VALUE events: keep if confidence >= high_value_threshold (0.40)
        MEDIUM_VALUE events: keep if confidence >= medium_value_threshold (0.55)
        SUPPRESS_ALWAYS events: always discard
        Everything else (unknown labels): treat as MEDIUM_VALUE

        Returns filtered list.
        """
        kept = []
        stats = {"high": 0, "medium": 0, "suppressed": 0, "below_threshold": 0}

        for event in events:
            label = event.label

            # Always suppress
            if label in self.suppress_set:
                stats["suppressed"] += 1
                continue

            # High value — lower threshold
            if label in self.high_value_set:
                if event.confidence >= self.high_value_threshold:
                    kept.append(event)
                    stats["high"] += 1
                else:
                    stats["below_threshold"] += 1
                continue

            # Medium value — higher threshold
            if label in MEDIUM_VALUE_CC_LABELS:
                if event.confidence >= self.medium_value_threshold:
                    kept.append(event)
                    stats["medium"] += 1
                else:
                    stats["below_threshold"] += 1
                continue

            # Unknown label: apply medium threshold
            if event.confidence >= self.medium_value_threshold:
                kept.append(event)
                stats["medium"] += 1
            else:
                stats["below_threshold"] += 1

        logger.info(
            f"Stage A (confidence filtering): "
            f"{len(events)} → {len(kept)} events kept\n"
            f"  high_value={stats['high']}, medium_value={stats['medium']}, "
            f"suppressed={stats['suppressed']}, "
            f"below_threshold={stats['below_threshold']}"
        )
        return kept


    # ─────────────────────────────────────────────────────────────────────
    # STAGE B: Window Merging
    # ─────────────────────────────────────────────────────────────────────

    def _merge_overlapping_windows(self, events: list) -> list:
        """
        Merge consecutive events with the same label that overlap or
        are within merge_gap_seconds of each other.

        Algorithm:
          1. Group events by label
          2. Within each group, sort by start_time
          3. Walk through sorted events:
             If current event overlaps or is within gap of previous:
               extend previous event's end_time
               update peak confidence
               increment window_count
             Else:
               start a new merged event

        Returns list of FilteredEvent objects.
        """
        if not events:
            return []

        # Group by label
        # defaultdict(list) creates an empty list for any new key
        by_label = defaultdict(list)
        for event in events:
            by_label[event.label].append(event)

        merged_events = []

        for label, label_events in by_label.items():
            # Sort by start time
            sorted_events = sorted(label_events, key=lambda e: e.start_time)

            # Initialize the first merge group
            current_start = sorted_events[0].start_time
            current_end = sorted_events[0].end_time
            current_confidences = [sorted_events[0].confidence]
            current_window_count = 1

            for event in sorted_events[1:]:
                # Check if this event overlaps with or is close to current group
                # "close" means: event starts within merge_gap_seconds after
                # the current group ends
                gap = event.start_time - current_end

                if gap <= self.merge_gap_seconds:
                    # Merge: extend the current group
                    current_end = max(current_end, event.end_time)
                    current_confidences.append(event.confidence)
                    current_window_count += 1
                else:
                    # Gap is too large — finalize current group, start new one
                    merged_events.append(FilteredEvent(
                        label=label,
                        cc_label=None,  # populated in Phase 11
                        start_time=current_start,
                        end_time=current_end,
                        peak_confidence=max(current_confidences),
                        avg_confidence=float(np.mean(current_confidences)),
                        window_count=current_window_count,
                        filter_reason="pending"  # set in Stage C
                    ))
                    # Start new group
                    current_start = event.start_time
                    current_end = event.end_time
                    current_confidences = [event.confidence]
                    current_window_count = 1

            # Don't forget the last group
            merged_events.append(FilteredEvent(
                label=label,
                cc_label=None,
                start_time=current_start,
                end_time=current_end,
                peak_confidence=max(current_confidences),
                avg_confidence=float(np.mean(current_confidences)),
                window_count=current_window_count,
                filter_reason="pending"
            ))

        # Sort final list by start_time for clean output
        merged_events.sort(key=lambda e: e.start_time)

        logger.info(
            f"Stage B (window merging): "
            f"{len(events)} → {len(merged_events)} merged events"
        )
        return merged_events


    # ─────────────────────────────────────────────────────────────────────
    # STAGE C: CC Relevance Filtering + Music Onset Detection
    # ─────────────────────────────────────────────────────────────────────

    def _detect_music_onsets(self, raw_events: list) -> list:
        """
        Scan raw (pre-merge) Music events chronologically and identify
        only the onset moments — where music transitions from absent to present.

        Args:
            raw_events: The original unmerged DetectedEvent list from PANNs.
                        We need the raw list (not merged) to see the
                        frame-by-frame confidence timeline.

        Returns:
            List of FilteredEvent objects representing music onsets only.
        """
        # Collect all music-related detections in time order
        # We look at any label in MUSIC_LABELS
        music_detections = []
        for event in raw_events:
            if event.label in MUSIC_LABELS or event.label == "Music":
                music_detections.append({
                    "time": event.start_time,
                    "confidence": event.confidence,
                    "label": event.label
                })

        if not music_detections:
            return []

        # Sort by time
        music_detections.sort(key=lambda x: x["time"])

        # Build a confidence timeline: for each 1-second step,
        # what was the max music confidence?
        if not music_detections:
            return []

        max_time = max(d["time"] for d in music_detections)
        # Create a dict: second → max music confidence at that second
        timeline = {}
        for d in music_detections:
            t = int(d["time"])
            timeline[t] = max(timeline.get(t, 0.0), d["confidence"])

        # Walk the timeline, detect onsets
        onset_events = []
        last_music_end_time = -999.0  # time when music last dropped below threshold
        music_is_active = False

        all_times = sorted(timeline.keys())

        for t in all_times:
            conf = timeline[t]
            was_active = music_is_active

            if conf >= self.music_onset_threshold:
                music_is_active = True
            elif conf < self.music_silence_threshold:
                music_is_active = False
                last_music_end_time = float(t)
            # Between thresholds: maintain current state (hysteresis)
            # This prevents rapid toggling when confidence hovers near threshold

            # Onset condition:
            # Music just became active AND either:
            #   a) it was previously inactive, OR
            #   b) enough silence has passed since it last ended
            if (music_is_active and not was_active and
                    (t - last_music_end_time) >= self.min_music_silence_gap):

                onset_events.append(FilteredEvent(
                    label="Music",
                    cc_label=None,
                    start_time=float(t),
                    end_time=float(t) + 2.0,  # 2s window for the onset
                    peak_confidence=conf,
                    avg_confidence=conf,
                    window_count=1,
                    filter_reason="music_onset"
                ))
                logger.info(
                    f"Music onset detected at {t}s "
                    f"(conf={conf:.3f}, "
                    f"silence since {last_music_end_time:.0f}s)"
                )

        logger.info(f"Music onset detection: {len(onset_events)} onsets found")
        return onset_events


    def _apply_relevance_filter(
        self,
        merged_events: list,
        raw_events: list
    ) -> list:
        """
        Apply CC relevance classification to merged events.
        Also runs music onset detection and injects onset events.

        For each merged event:
          - HIGH_VALUE labels → keep, set filter_reason="high_value"
          - MEDIUM_VALUE labels → keep, set filter_reason="medium_value"
          - MUSIC_LABELS → suppress (handled by onset detection instead)
          - SUPPRESS_ALWAYS → discard (should already be gone from Stage A,
            but this is a safety net)

        Then append music onset events from _detect_music_onsets().
        """
        final_events = []
        suppressed_music_count = 0
        kept_count = 0

        for event in merged_events:
            label = event.label

            # Safety net: suppress if in suppress set
            if label in self.suppress_set:
                continue

            # Suppress sustained music (onset detection handles Music separately)
            if label in MUSIC_LABELS and self.suppress_sustained_music:
                suppressed_music_count += 1
                continue

            # High value: tag and keep
            if label in self.high_value_set:
                event.filter_reason = "high_value"
                final_events.append(event)
                kept_count += 1
                continue

            # Medium value: tag and keep
            if label in MEDIUM_VALUE_CC_LABELS:
                event.filter_reason = "medium_value"
                final_events.append(event)
                kept_count += 1
                continue

            # Unknown label: keep with "unknown" reason for human review
            event.filter_reason = "unknown"
            final_events.append(event)
            kept_count += 1

        logger.info(
            f"Stage C (relevance filter): "
            f"kept={kept_count}, "
            f"suppressed_music={suppressed_music_count}"
        )

        # Add music onset events
        if self.suppress_sustained_music:
            onset_events = self._detect_music_onsets(raw_events)
            final_events.extend(onset_events)
            logger.info(f"Added {len(onset_events)} music onset events")

        # Re-sort by start_time after adding onsets
        final_events.sort(key=lambda e: e.start_time)

        return final_events


    # ─────────────────────────────────────────────────────────────────────
    # SECTION: Main Entry Point
    # ─────────────────────────────────────────────────────────────────────

    def filter(self, detection_result) -> list:
        """
        Run the full three-stage filtering pipeline.

        Args:
            detection_result: DetectionResult from SoundDetector.detect()

        Returns:
            List of FilteredEvent objects — the clean candidate CC events
            ready for visual confirmation in Phase 6+.
        """
        raw_events = detection_result.events

        logger.info("=" * 60)
        logger.info(
            f"Starting audio event filtering: "
            f"{len(raw_events)} raw events"
        )
        logger.info("=" * 60)

        # Stage A: Confidence threshold
        stage_a = self._apply_confidence_threshold(raw_events)

        # Stage B: Merge overlapping windows
        stage_b = self._merge_overlapping_windows(stage_a)

        # Stage C: Relevance filter + music onset detection
        # Note: we pass raw_events to Stage C for music onset detection
        # (we need the full unmerged timeline to detect transitions)
        stage_c = self._apply_relevance_filter(stage_b, raw_events)

        logger.info("=" * 60)
        logger.info(
            f"Filtering complete: "
            f"{len(raw_events)} → {len(stage_c)} candidate CC events"
        )

        if stage_c:
            logger.info("Final candidate events:")
            for e in stage_c:
                logger.info(
                    f"  [{e.filter_reason}] {e.label} "
                    f"{e.start_time:.1f}s→{e.end_time:.1f}s "
                    f"(conf={e.peak_confidence:.3f}, "
                    f"merged {e.window_count}w)"
                )
        else:
            logger.warning(
                "No events passed filtering. Consider lowering thresholds "
                "or checking if the video has non-speech audio events."
            )

        logger.info("=" * 60)
        return stage_c


    # ─────────────────────────────────────────────────────────────────────
    # SECTION: Summary Report
    # ─────────────────────────────────────────────────────────────────────

    def summarize(self, filtered_events: list) -> str:
        """
        Generate a human-readable summary of filtered events.
        Useful for logging and debugging.
        """
        if not filtered_events:
            return "No events passed filtering."

        lines = [
            f"{'─' * 60}",
            f"FILTERED EVENT SUMMARY ({len(filtered_events)} events)",
            f"{'─' * 60}",
        ]

        for i, e in enumerate(filtered_events, 1):
            mins = int(e.start_time // 60)
            secs = e.start_time % 60
            lines.append(
                f"{i:3d}. [{mins:02d}:{secs:05.2f}] "
                f"{e.label:<35s} "
                f"conf={e.peak_confidence:.3f}  "
                f"({e.filter_reason})"
            )

        lines.append(f"{'─' * 60}")
        return "\n".join(lines)