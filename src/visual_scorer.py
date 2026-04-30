# =============================================================================
# STAGE 3 (Part 4): Visual Reaction Confidence Scoring
# =============================================================================
# Combines FaceReactionResult (Phase 7) and PoseReactionResult (Phase 8)
# into a single visual_confidence score per event.
#
# Also handles:
#   - Detection-aware combination (neutral 0.5 ≠ real evidence of 0.5)
#   - Event deduplication (Bow-wow + Animal at same timestamp → keep one)
#   - Per-event scoring summary for the CC decision engine
# =============================================================================

import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("visual_scorer")


# =============================================================================
# SECTION: AudioSet Specificity Map
# =============================================================================
# More specific labels score lower (more specific = preferred when deduplicating).
# This is a partial map of the AudioSet hierarchy for labels likely to appear
# in our pipeline. Labels not in this map get specificity=50 (medium).
#
# Lower number = more specific (prefer this label in deduplication).
# Higher number = more general (deprioritize in deduplication).

LABEL_SPECIFICITY = {
    # Animals — specific to general
    "Bow-wow": 10,
    "Bark": 11,
    "Cat": 15,
    "Meow": 16,
    "Bird": 20,
    "Chirp, tweet": 21,
    "Animal": 90,        # very general

    # Vehicles — specific to general
    "Honk": 10,
    "Car alarm": 11,
    "Skidding": 12,
    "Engine starting": 13,
    "Car": 30,
    "Motorcycle": 30,
    "Train": 30,
    "Vehicle": 90,        # very general

    # Impact sounds — all roughly equally specific
    "Gunshot, gunfire": 10,
    "Explosion": 10,
    "Glass": 15,
    "Breaking": 16,
    "Bang": 20,
    "Slam": 20,
    "Crash": 20,
    "Smash, crash": 21,

    # Human reactions
    "Laughter": 10,
    "Crying, sobbing": 10,
    "Screaming": 10,
    "Applause": 10,
    "Crowd": 50,

    # Mechanisms
    "Ratchet, pawl": 10,
    "Chopping (food)": 10,
    "Mechanisms": 40,

    # Music
    "Music": 80,
    "Music of Bollywood": 30,
    "Tabla music": 20,
}


# =============================================================================
# SECTION: Data Container
# =============================================================================

@dataclass
class ScoredEvent:
    """
    A fully scored event ready for the CC decision engine (Phase 10).

    This is the unified output of the entire visual pipeline (Phases 6–9).
    It combines the filtered audio event with its visual confidence scores.

    Fields:
        filtered_event:      The FilteredEvent from Phase 5
                             (has label, timestamps, audio confidence,
                              filter_reason)
        face_score:          Face reaction score from Phase 7 (0.0–1.0)
        pose_score:          Pose reaction score from Phase 8 (0.0–1.0)
        visual_confidence:   Combined visual signal (0.0–1.0)
                             This is what Phase 10 uses.
        face_detected:       Whether a face was found in the event frames
        pose_detected:       Whether a pose was found in the event frames
        combination_method:  How visual_confidence was computed
                             ("dual", "face_only", "pose_only", "none")
        is_duplicate:        True if this event was marked as a duplicate
                             of another event at the same timestamp
    """
    filtered_event: object      # FilteredEvent from event_filter.py
    face_score: float
    pose_score: float
    visual_confidence: float
    face_detected: bool
    pose_detected: bool
    combination_method: str
    is_duplicate: bool = False

    @property
    def label(self) -> str:
        return self.filtered_event.label

    @property
    def start_time(self) -> float:
        return self.filtered_event.start_time

    @property
    def end_time(self) -> float:
        return self.filtered_event.end_time

    @property
    def audio_confidence(self) -> float:
        return self.filtered_event.peak_confidence

    @property
    def filter_reason(self) -> str:
        return self.filtered_event.filter_reason

    def __repr__(self):
        return (
            f"ScoredEvent({self.label!r}, "
            f"{self.start_time:.1f}s→{self.end_time:.1f}s, "
            f"audio={self.audio_confidence:.3f}, "
            f"visual={self.visual_confidence:.3f}, "
            f"method={self.combination_method!r}, "
            f"dup={self.is_duplicate})"
        )


# =============================================================================
# SECTION: Core Visual Scorer
# =============================================================================

class VisualScorer:
    """
    Combines face and pose reaction scores into a single visual_confidence
    value per event, then deduplicates events at the same timestamp.
    """

    def __init__(
        self,
        face_weight: float = 0.6,
        pose_weight: float = 0.4,
        single_signal_discount: float = 0.9,
        duplicate_time_threshold: float = 1.0,
    ):
        """
        Args:
            face_weight:             Weight for face score in dual-signal
                                     combination. face + pose weights should
                                     conceptually sum to 1.0.
            pose_weight:             Weight for pose score in dual-signal
                                     combination.
            single_signal_discount:  Multiplier applied when only one of
                                     face/pose was detected.
                                     0.9 = 10% discount for single signal.
                                     Encodes: dual confirmation > single.
            duplicate_time_threshold: Events within this many seconds of
                                      each other are considered potential
                                      duplicates of the same physical event.
                                      Default 1.0s = within the same second.
        """
        self.face_weight = face_weight
        self.pose_weight = pose_weight
        self.single_signal_discount = single_signal_discount
        self.duplicate_time_threshold = duplicate_time_threshold

        logger.info(
            f"VisualScorer initialized: "
            f"face_weight={face_weight}, pose_weight={pose_weight}, "
            f"single_signal_discount={single_signal_discount}, "
            f"dup_threshold={duplicate_time_threshold}s"
        )


    # ─────────────────────────────────────────────────────────────────────
    # SECTION: Single Event Scoring
    # ─────────────────────────────────────────────────────────────────────

    def _combine_scores(
        self,
        face_score: float,
        pose_score: float,
        face_detected: bool,
        pose_detected: bool
    ) -> tuple:
        """
        Combine face and pose scores into a single visual_confidence.

        Returns (visual_confidence, combination_method).

        The combination_method string documents which signals contributed,
        useful for debugging and editor review.

        Logic:
          Both detected:  weighted average of face and pose scores
          Face only:      face score × single_signal_discount
          Pose only:      pose score × single_signal_discount
          Neither:        0.5 (no visual evidence)

        Why not always use the weighted average?
        If pose_detected=False, pose_score=0.5 (neutral placeholder).
        Including 0.5 in the weighted average artificially pulls the
        visual_confidence toward 0.5 even when the face score is 0.8.
        The detection-aware logic prevents this dilution.
        """
        if face_detected and pose_detected:
            # Both signals are real evidence — standard weighted combination
            visual_conf = (
                self.face_weight * face_score +
                self.pose_weight * pose_score
            )
            method = "dual"

        elif face_detected and not pose_detected:
            # Only face signal is real. Pose was 0.5 placeholder.
            # Use face score directly, with slight discount for
            # single-signal uncertainty.
            visual_conf = face_score * self.single_signal_discount
            method = "face_only"

        elif pose_detected and not face_detected:
            # Only pose signal is real. Face was 0.5 placeholder.
            visual_conf = pose_score * self.single_signal_discount
            method = "pose_only"

        else:
            # Neither detected — no visual evidence at all.
            # Return exactly 0.5 (neutral).
            visual_conf = 0.5
            method = "none"

        # Clamp to [0, 1] as a safety measure
        visual_conf = float(np.clip(visual_conf, 0.0, 1.0))

        return visual_conf, method


    # ─────────────────────────────────────────────────────────────────────
    # SECTION: Event Deduplication
    # ─────────────────────────────────────────────────────────────────────

    def _deduplicate_events(self, scored_events: list) -> list:
        """
        Mark duplicate events — events at the same timestamp that
        describe the same physical sound at different levels of specificity.

        Algorithm:
          1. Sort events by start_time
          2. Group events whose start_times are within duplicate_time_threshold
          3. Within each group, keep the event with the lowest specificity
             score (= most specific label)
          4. If specificity is equal, keep the one with higher audio confidence
          5. Mark all others in the group as is_duplicate=True

        Returns the same list with is_duplicate field set correctly.
        Non-duplicate events are unchanged.
        """
        if len(scored_events) <= 1:
            return scored_events

        # Sort by start_time for grouping
        sorted_events = sorted(scored_events, key=lambda e: e.start_time)

        # Find groups of events at the same timestamp
        # A "group" is events whose start_times are all within
        # duplicate_time_threshold of each other
        groups = []
        current_group = [sorted_events[0]]

        for event in sorted_events[1:]:
            # Check if this event is within threshold of the group's first event
            group_start = current_group[0].start_time
            if abs(event.start_time - group_start) <= self.duplicate_time_threshold:
                current_group.append(event)
            else:
                groups.append(current_group)
                current_group = [event]
        groups.append(current_group)

        # Process each group
        result = []
        total_duplicates = 0

        for group in groups:
            if len(group) == 1:
                # Single event — no deduplication needed
                result.append(group[0])
                continue

            # Multiple events at the same timestamp
            # Find the "best" event to keep:
            # 1. Most specific label (lowest LABEL_SPECIFICITY value)
            # 2. Tie-break: higher audio confidence

            def event_priority(e):
                specificity = LABEL_SPECIFICITY.get(e.label, 50)
                # Lower specificity score = more specific = sort first
                # Negate audio confidence so higher confidence sorts first
                return (specificity, -e.audio_confidence)

            group_sorted = sorted(group, key=event_priority)
            keeper = group_sorted[0]
            duplicates = group_sorted[1:]

            keeper.is_duplicate = False
            result.append(keeper)

            for dup in duplicates:
                dup.is_duplicate = True
                result.append(dup)
                total_duplicates += 1
                logger.info(
                    f"Duplicate marked: '{dup.label}' @ {dup.start_time:.1f}s "
                    f"(kept '{keeper.label}' — more specific or higher conf)"
                )

        if total_duplicates > 0:
            logger.info(
                f"Deduplication: {total_duplicates} duplicate(s) marked "
                f"across {len(scored_events)} events"
            )

        # Re-sort by start_time (deduplication may have reordered)
        result.sort(key=lambda e: e.start_time)
        return result


    # ─────────────────────────────────────────────────────────────────────
    # SECTION: Main Entry Point
    # ─────────────────────────────────────────────────────────────────────

    def score(
        self,
        filtered_events: list,
        face_results: list,
        pose_results: list
    ) -> list:
        """
        Combine face and pose results into ScoredEvent objects.

        Args:
            filtered_events: list[FilteredEvent] from Phase 5
            face_results:    list[FaceReactionResult] from Phase 7
                             Must be same length and order as filtered_events
            pose_results:    list[PoseReactionResult] from Phase 8
                             Must be same length and order as filtered_events

        Returns:
            list[ScoredEvent] — one per event, with visual_confidence set.
            Events marked is_duplicate=True are still in the list but will
            be filtered out by the CC decision engine in Phase 10.
        """
        # Validate alignment
        assert len(filtered_events) == len(face_results) == len(pose_results), (
            f"Length mismatch: "
            f"filtered_events={len(filtered_events)}, "
            f"face_results={len(face_results)}, "
            f"pose_results={len(pose_results)}"
        )

        logger.info("=" * 60)
        logger.info(
            f"Starting visual confidence scoring: "
            f"{len(filtered_events)} events"
        )
        logger.info("=" * 60)

        scored_events = []

        for i, (event, face_r, pose_r) in enumerate(
            zip(filtered_events, face_results, pose_results)
        ):
            # Combine face and pose scores
            visual_conf, method = self._combine_scores(
                face_score=face_r.face_reaction_score,
                pose_score=pose_r.pose_reaction_score,
                face_detected=face_r.faces_detected,
                pose_detected=pose_r.pose_detected
            )

            scored = ScoredEvent(
                filtered_event=event,
                face_score=face_r.face_reaction_score,
                pose_score=pose_r.pose_reaction_score,
                visual_confidence=visual_conf,
                face_detected=face_r.faces_detected,
                pose_detected=pose_r.pose_detected,
                combination_method=method,
                is_duplicate=False
            )

            scored_events.append(scored)

            logger.info(
                f"[{i+1}/{len(filtered_events)}] "
                f"'{event.label}' @ {event.start_time:.1f}s: "
                f"face={face_r.face_reaction_score:.3f} "
                f"({'det' if face_r.faces_detected else 'no-det'}), "
                f"pose={pose_r.pose_reaction_score:.3f} "
                f"({'det' if pose_r.pose_detected else 'no-det'}) "
                f"→ visual_conf={visual_conf:.3f} [{method}]"
            )

        # Deduplicate events at the same timestamp
        scored_events = self._deduplicate_events(scored_events)

        # Summary
        non_dupes = [e for e in scored_events if not e.is_duplicate]
        high_visual = [
            e for e in non_dupes if e.visual_confidence > 0.6
        ]
        no_visual = [
            e for e in non_dupes if e.combination_method == "none"
        ]

        logger.info("=" * 60)
        logger.info(f"Visual scoring complete:")
        logger.info(f"  Total events:          {len(scored_events)}")
        logger.info(f"  Unique events:         {len(non_dupes)}")
        logger.info(
            f"  Duplicates marked:     "
            f"{len(scored_events) - len(non_dupes)}"
        )
        logger.info(
            f"  High visual conf>0.6:  {len(high_visual)}"
        )
        logger.info(
            f"  No visual signal:      {len(no_visual)}"
        )
        logger.info("=" * 60)

        return scored_events


    # ─────────────────────────────────────────────────────────────────────
    # SECTION: Summary Display
    # ─────────────────────────────────────────────────────────────────────

    def summarize(self, scored_events: list) -> str:
        """Human-readable summary of all scored events."""
        lines = [
            f"{'─' * 72}",
            f"VISUAL CONFIDENCE SUMMARY ({len(scored_events)} events)",
            f"{'─' * 72}",
            f"{'Event':<25} {'Time':>6} {'Audio':>6} "
            f"{'Face':>6} {'Pose':>6} {'Visual':>7} "
            f"{'Method':<10} {'Dup':>4}",
            f"{'─' * 72}",
        ]

        for e in scored_events:
            mins = int(e.start_time // 60)
            secs = e.start_time % 60
            dup_mark = "YES" if e.is_duplicate else "-"
            lines.append(
                f"{e.label:<25} "
                f"{mins:02d}:{secs:04.1f} "
                f"{e.audio_confidence:>6.3f} "
                f"{e.face_score:>6.3f} "
                f"{e.pose_score:>6.3f} "
                f"{e.visual_confidence:>7.3f} "
                f"{e.combination_method:<10} "
                f"{dup_mark:>4}"
            )

        lines.append(f"{'─' * 72}")

        non_dupes = [e for e in scored_events if not e.is_duplicate]
        lines.append(
            f"Unique events: {len(non_dupes)} "
            f"| Duplicates: {len(scored_events) - len(non_dupes)}"
        )
        return "\n".join(lines)