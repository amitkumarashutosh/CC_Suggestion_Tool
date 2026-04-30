# =============================================================================
# STAGE 4: CC Decision Engine
# =============================================================================
# Takes ScoredEvents from visual_scorer.py and makes the final yes/no
# decision: does this event deserve a closed caption annotation?
#
# Combines audio_confidence + visual_confidence with configurable weights,
# applies event-type-aware thresholds, and returns CCDecision objects
# — the final output before SRT generation in Phase 12.
# =============================================================================

import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("decision_engine")


# =============================================================================
# SECTION: Data Container
# =============================================================================

@dataclass
class CCDecision:
    """
    The final decision for a single audio event.

    Fields:
        label:             AudioSet class name (e.g. "Bow-wow")
        start_time:        Event start in seconds
        end_time:          Event end in seconds
        audio_confidence:  PANNs confidence score (0.0–1.0)
        visual_confidence: Combined visual signal (0.0–1.0)
        cc_score:          Final combined score used for decision
        threshold_used:    The threshold this event was compared against
        accepted:          True → include in SRT. False → discard.
        rejection_reason:  Why rejected (None if accepted)
        filter_reason:     "high_value", "medium_value", "music_onset"
        combination_method: "dual", "face_only", "pose_only", "none"
        cc_label:          Human-readable CC text e.g. "[Dog Barking]"
                           Populated by Phase 11 (label_generator.py)
    """
    label: str
    start_time: float
    end_time: float
    audio_confidence: float
    visual_confidence: float
    cc_score: float
    threshold_used: float
    accepted: bool
    rejection_reason: Optional[str]
    filter_reason: str
    combination_method: str
    cc_label: Optional[str] = None  # set by Phase 11

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def timestamp_str(self) -> str:
        """Human-readable timestamp: MM:SS.s"""
        mins = int(self.start_time // 60)
        secs = self.start_time % 60
        return f"{mins:02d}:{secs:05.2f}"

    def __repr__(self):
        status = "ACCEPT" if self.accepted else f"REJECT({self.rejection_reason})"
        return (
            f"CCDecision({self.label!r} @ {self.timestamp_str}, "
            f"score={self.cc_score:.3f}/{self.threshold_used:.2f}, "
            f"{status})"
        )


# =============================================================================
# SECTION: Core Decision Engine
# =============================================================================

class CCDecisionEngine:
    """
    Applies weighted combination of audio and visual confidence,
    then compares to event-type-aware thresholds to produce
    final CC accept/reject decisions.
    """

    def __init__(
        self,
        audio_weight: float = 0.65,
        visual_weight: float = 0.35,
        cc_threshold: float = 0.60,

        # Per-event-type threshold adjustments
        high_value_threshold_delta: float = -0.05,
        music_onset_threshold_delta: float = -0.10,
        medium_value_threshold_delta: float = 0.00,
        unknown_threshold_delta: float = +0.05,

        # High-value boost
        high_value_boost: float = 0.05,
        high_value_boost_min_audio: float = 0.45,

        # Visual neutrality handling
        # If visual_confidence is exactly neutral (no detection),
        # we reduce its weight so it doesn't dilute the audio signal
        neutral_visual_weight_reduction: float = 0.5,
    ):
        """
        Args:
            audio_weight:      Weight for audio_confidence in cc_score.
            visual_weight:     Weight for visual_confidence in cc_score.
            cc_threshold:      Base threshold for CC acceptance.
                               Events with cc_score >= threshold are accepted.
            high_value_threshold_delta:  Threshold adjustment for HIGH_VALUE
                               events. Negative = easier to pass.
            music_onset_threshold_delta: Threshold adjustment for music onsets.
            medium_value_threshold_delta: Threshold for MEDIUM_VALUE events.
            unknown_threshold_delta:     Threshold for unknown label events.
            high_value_boost:  Score boost added to HIGH_VALUE events above
                               high_value_boost_min_audio confidence.
            high_value_boost_min_audio:  Minimum audio confidence to qualify
                               for the high_value_boost.
            neutral_visual_weight_reduction: When visual_confidence=0.5
                               (no detection), reduce its effective weight
                               by this fraction. 0.5 = use half the visual
                               weight, reassign other half to audio.
                               This prevents neutral visual from diluting
                               strong audio confidence.
        """
        self.audio_weight  = audio_weight
        self.visual_weight = visual_weight
        self.cc_threshold  = cc_threshold

        self.threshold_deltas = {
            "high_value":   high_value_threshold_delta,
            "music_onset":  music_onset_threshold_delta,
            "medium_value": medium_value_threshold_delta,
            "unknown":      unknown_threshold_delta,
        }

        self.high_value_boost         = high_value_boost
        self.high_value_boost_min_audio = high_value_boost_min_audio
        self.neutral_visual_weight_reduction = neutral_visual_weight_reduction

        logger.info(
            f"CCDecisionEngine initialized:\n"
            f"  audio_weight={audio_weight}, visual_weight={visual_weight}\n"
            f"  base_threshold={cc_threshold}\n"
            f"  threshold_deltas={self.threshold_deltas}\n"
            f"  high_value_boost={high_value_boost} "
            f"(min_audio={high_value_boost_min_audio})"
        )


    # ─────────────────────────────────────────────────────────────────────
    # SECTION: Score Computation
    # ─────────────────────────────────────────────────────────────────────

    def _compute_cc_score(
        self,
        audio_confidence: float,
        visual_confidence: float,
        combination_method: str,
        filter_reason: str
    ) -> float:
        """
        Compute the combined cc_score from audio and visual confidence.

        Handles three cases:
          1. Visual detected (dual/face_only/pose_only):
             Standard weighted average.
          2. Visual not detected (none/neutral 0.5):
             Reduce visual weight, redistribute to audio.
             Prevents neutral 0.5 from dragging down strong audio.
          3. High-value boost:
             Add small bonus score for inherently CC-worthy event types.
        """
        # Case 2: No visual detection — adjust weights
        if combination_method == "none":
            # Visual signal is 0.5 (neutral placeholder, not real evidence).
            # Reduce its weight and give that weight back to audio.
            reduction = self.neutral_visual_weight_reduction
            effective_visual_weight = self.visual_weight * (1.0 - reduction)
            effective_audio_weight  = self.audio_weight  + (
                self.visual_weight * reduction
            )
            # With default params:
            # effective_audio_weight  = 0.65 + (0.35 × 0.5) = 0.825
            # effective_visual_weight = 0.35 × 0.5           = 0.175
            # This ensures audio signal dominates when visual is absent
        else:
            effective_audio_weight  = self.audio_weight
            effective_visual_weight = self.visual_weight

        # Weighted combination
        cc_score = (
            effective_audio_weight  * audio_confidence +
            effective_visual_weight * visual_confidence
        )

        # Case 3: High-value boost
        if (filter_reason == "high_value"
                and audio_confidence >= self.high_value_boost_min_audio):
            cc_score += self.high_value_boost
            logger.debug(
                f"High-value boost applied: +{self.high_value_boost} "
                f"(audio={audio_confidence:.3f} >= "
                f"{self.high_value_boost_min_audio})"
            )

        # Clamp to [0, 1]
        return float(np.clip(cc_score, 0.0, 1.0))


    def _get_threshold(self, filter_reason: str) -> float:
        """
        Get the effective threshold for this event type.

        Returns base threshold adjusted by the per-type delta.
        """
        delta = self.threshold_deltas.get(filter_reason, 0.0)
        threshold = self.cc_threshold + delta
        # Clamp to reasonable range [0.3, 0.9]
        return float(np.clip(threshold, 0.3, 0.9))


    # ─────────────────────────────────────────────────────────────────────
    # SECTION: Single Event Decision
    # ─────────────────────────────────────────────────────────────────────

    def _decide_event(self, scored_event) -> CCDecision:
        """
        Make the CC decision for a single ScoredEvent.

        Returns a CCDecision with accepted=True or False
        and a rejection_reason if rejected.
        """
        # Skip duplicates immediately — they are never accepted
        if scored_event.is_duplicate:
            return CCDecision(
                label=scored_event.label,
                start_time=scored_event.start_time,
                end_time=scored_event.end_time,
                audio_confidence=scored_event.audio_confidence,
                visual_confidence=scored_event.visual_confidence,
                cc_score=0.0,
                threshold_used=0.0,
                accepted=False,
                rejection_reason="duplicate",
                filter_reason=scored_event.filter_reason,
                combination_method=scored_event.combination_method,
                cc_label=None
            )

        # Compute the combined cc_score
        cc_score = self._compute_cc_score(
            audio_confidence=scored_event.audio_confidence,
            visual_confidence=scored_event.visual_confidence,
            combination_method=scored_event.combination_method,
            filter_reason=scored_event.filter_reason
        )

        # Get the effective threshold for this event type
        threshold = self._get_threshold(scored_event.filter_reason)

        # Make the decision
        if cc_score >= threshold:
            accepted = True
            rejection_reason = None
        else:
            accepted = False
            # Provide a specific reason for rejection to aid debugging
            if cc_score < threshold * 0.7:
                rejection_reason = "score_too_low"
            elif scored_event.audio_confidence < 0.45:
                rejection_reason = "audio_confidence_too_low"
            elif scored_event.visual_confidence < 0.52:
                rejection_reason = "weak_visual_confirmation"
            else:
                rejection_reason = "below_threshold"

        return CCDecision(
            label=scored_event.label,
            start_time=scored_event.start_time,
            end_time=scored_event.end_time,
            audio_confidence=scored_event.audio_confidence,
            visual_confidence=scored_event.visual_confidence,
            cc_score=cc_score,
            threshold_used=threshold,
            accepted=accepted,
            rejection_reason=rejection_reason,
            filter_reason=scored_event.filter_reason,
            combination_method=scored_event.combination_method,
            cc_label=None  # populated by Phase 11
        )


    # ─────────────────────────────────────────────────────────────────────
    # SECTION: Main Entry Point
    # ─────────────────────────────────────────────────────────────────────

    def decide(self, scored_events: list) -> list:
        """
        Make CC decisions for all ScoredEvents.

        Args:
            scored_events: list[ScoredEvent] from visual_scorer.py

        Returns:
            list[CCDecision] — ALL events (accepted and rejected).
            Filter for accepted=True to get CC annotations.
            Keep all for debugging and threshold tuning.
        """
        logger.info("=" * 60)
        logger.info(
            f"Starting CC decision engine: "
            f"{len(scored_events)} candidate events"
        )
        logger.info("=" * 60)

        decisions = []
        for event in scored_events:
            decision = self._decide_event(event)
            decisions.append(decision)

            status = "✓ ACCEPT" if decision.accepted else f"✗ REJECT"
            reason = (
                f"(reason: {decision.rejection_reason})"
                if not decision.accepted else ""
            )
            logger.info(
                f"  {status}: '{decision.label}' @ {decision.timestamp_str} "
                f"score={decision.cc_score:.3f} "
                f"threshold={decision.threshold_used:.2f} "
                f"{reason}"
            )

        # Summary
        accepted = [d for d in decisions if d.accepted]
        rejected = [d for d in decisions if not d.accepted]

        logger.info("=" * 60)
        logger.info(f"Decision engine complete:")
        logger.info(f"  ACCEPTED: {len(accepted)} events → will become CC annotations")
        logger.info(f"  REJECTED: {len(rejected)} events → discarded")

        if accepted:
            logger.info("Accepted events:")
            for d in accepted:
                logger.info(
                    f"    [{d.filter_reason}] '{d.label}' "
                    f"@ {d.timestamp_str} "
                    f"(audio={d.audio_confidence:.3f}, "
                    f"visual={d.visual_confidence:.3f}, "
                    f"cc_score={d.cc_score:.3f})"
                )

        if rejected:
            logger.info("Rejected events:")
            for d in rejected:
                logger.info(
                    f"    [{d.rejection_reason}] '{d.label}' "
                    f"@ {d.timestamp_str} "
                    f"(cc_score={d.cc_score:.3f} "
                    f"< threshold={d.threshold_used:.2f})"
                )

        logger.info("=" * 60)
        return decisions


    # ─────────────────────────────────────────────────────────────────────
    # SECTION: Summary and Threshold Tuning Helper
    # ─────────────────────────────────────────────────────────────────────

    def summarize(self, decisions: list) -> str:
        """Human-readable decision summary."""
        accepted = [d for d in decisions if d.accepted]
        rejected = [d for d in decisions if not d.accepted]

        lines = [
            f"{'─' * 72}",
            f"CC DECISION SUMMARY",
            f"  Accepted: {len(accepted)}  |  Rejected: {len(rejected)}",
            f"{'─' * 72}",
            f"{'Event':<25} {'Time':>6} {'Audio':>6} "
            f"{'Visual':>7} {'Score':>6} {'Thresh':>7} {'Decision':>10}",
            f"{'─' * 72}",
        ]

        for d in decisions:
            mins = int(d.start_time // 60)
            secs = d.start_time % 60
            decision_str = (
                "✓ ACCEPT" if d.accepted
                else f"✗ {d.rejection_reason}"
            )
            lines.append(
                f"{d.label:<25} "
                f"{mins:02d}:{secs:04.1f} "
                f"{d.audio_confidence:>6.3f} "
                f"{d.visual_confidence:>7.3f} "
                f"{d.cc_score:>6.3f} "
                f"{d.threshold_used:>7.2f} "
                f"{decision_str:>10}"
            )

        lines.append(f"{'─' * 72}")
        return "\n".join(lines)


    def what_if_threshold(
        self,
        decisions: list,
        new_threshold: float
    ) -> str:
        """
        Show what would change if the threshold were different.
        Useful for threshold tuning without re-running the pipeline.

        Args:
            decisions:     list[CCDecision] from decide()
            new_threshold: hypothetical new base threshold

        Returns:
            String showing which events would flip accept/reject.
        """
        lines = [
            f"What-if analysis: threshold {self.cc_threshold:.2f} "
            f"→ {new_threshold:.2f}",
            f"{'─' * 50}",
        ]

        flips = 0
        for d in decisions:
            if d.rejection_reason == "duplicate":
                continue
            # Recompute with new threshold
            delta = self.threshold_deltas.get(d.filter_reason, 0.0)
            new_eff_threshold = float(np.clip(new_threshold + delta, 0.3, 0.9))
            new_accepted = d.cc_score >= new_eff_threshold

            if new_accepted != d.accepted:
                flips += 1
                old_str = "ACCEPT" if d.accepted else "REJECT"
                new_str = "ACCEPT" if new_accepted else "REJECT"
                lines.append(
                    f"  FLIP: '{d.label}' @ {d.timestamp_str} "
                    f"{old_str} → {new_str} "
                    f"(score={d.cc_score:.3f}, "
                    f"new_thresh={new_eff_threshold:.2f})"
                )

        if flips == 0:
            lines.append(
                f"  No changes — same {len([d for d in decisions if d.accepted])} "
                f"events accepted at threshold {new_threshold:.2f}"
            )

        lines.append(f"{'─' * 50}")
        return "\n".join(lines)