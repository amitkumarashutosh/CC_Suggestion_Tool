# src/label_generator.py
# =============================================================================
# STAGE 4 (Part 2): CC Label Auto-Generation
# =============================================================================
# Takes accepted CCDecision objects from decision_engine.py and populates
# the cc_label field with human-readable CC text.
#
# "Bow-wow"          → "[Dog Barking]"
# "Chopping (food)"  → "[Chopping Sounds]"
# "Music" (onset)    → "[Music]"
# "Vehicle"          → "[Vehicle]"
#
# Uses a rule-based label map for known AudioSet labels,
# with a fallback generator for unknown labels.
# =============================================================================

import re
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("label_generator")


# =============================================================================
# SECTION: Primary Label Map
# =============================================================================
# Maps AudioSet display_name → CC text shown to viewers.
#
# CC text rules (industry standard):
#   - Always bracketed: [Label Text]
#   - Title Case: first letter of each significant word capitalized
#   - 1-4 words maximum
#   - Describes the sound, not the source taxonomy
#   - Uses active/descriptive form where possible:
#     "Dog Barking" not "Dog Bark", "Glass Shattering" not "Glass"
#
# Organized by category for maintainability.
# Editors can extend this map via config.yaml in Phase 13.

LABEL_MAP = {

    # ── ANIMALS ──────────────────────────────────────────────────────────
    "Bow-wow":                          "[Dog Barking]",
    "Bark":                             "[Dog Barking]",
    "Dog":                              "[Dog]",
    "Yip":                              "[Dog Barking]",
    "Howl":                             "[Howling]",
    "Cat":                              "[Cat]",
    "Meow":                             "[Cat Meowing]",
    "Purr":                             "[Cat Purring]",
    "Bird":                             "[Bird]",
    "Chirp, tweet":                     "[Bird Chirping]",
    "Crow":                             "[Crow Cawing]",
    "Animal":                           "[Animal Sound]",
    "Insect":                           "[Insect Sound]",

    # ── IMPACT / COLLISION ───────────────────────────────────────────────
    "Gunshot, gunfire":                 "[Gunshot]",
    "Gunshot":                          "[Gunshot]",
    "Explosion":                        "[Explosion]",
    "Boom":                             "[Loud Boom]",
    "Glass":                            "[Glass Shattering]",
    "Shatter":                          "[Glass Shattering]",
    "Breaking":                         "[Breaking Sound]",
    "Bang":                             "[Bang]",
    "Slam":                             "[Door Slam]",
    "Crash":                            "[Crash]",
    "Smash, crash":                     "[Crash]",
    "Thud":                             "[Thud]",
    "Knock":                            "[Knocking]",
    "Tap":                              "[Tapping]",

    # ── HUMAN REACTIONS ──────────────────────────────────────────────────
    "Laughter":                         "[Laughter]",
    "Giggling":                         "[Laughter]",
    "Chuckling, kn chuckling":          "[Chuckling]",
    "Crying, sobbing":                  "[Crying]",
    "Whimper":                          "[Whimpering]",
    "Screaming":                        "[Screaming]",
    "Shout":                            "[Shouting]",
    "Applause":                         "[Applause]",
    "Clapping":                         "[Applause]",
    "Crowd":                            "[Crowd Noise]",
    "Cheering":                         "[Cheering]",
    "Booing":                           "[Booing]",
    "Whistling":                        "[Whistling]",
    "Cough":                            "[Coughing]",
    "Sneeze":                           "[Sneezing]",

    # ── ALARMS / SIGNALS ─────────────────────────────────────────────────
    "Telephone":                        "[Phone Ringing]",
    "Telephone bell ringing":           "[Phone Ringing]",
    "Ringtone":                         "[Phone Ringing]",
    "Alarm":                            "[Alarm]",
    "Siren":                            "[Siren]",
    "Doorbell":                         "[Doorbell]",
    "Bell":                             "[Bell]",
    "Smoke detector, smoke alarm":      "[Smoke Alarm]",
    "Alarm clock":                      "[Alarm Clock]",
    "Beep, bleep":                      "[Beeping]",

    # ── VEHICLES ─────────────────────────────────────────────────────────
    "Vehicle":                          "[Vehicle]",
    "Car":                              "[Car]",
    "Motorcycle":                       "[Motorcycle]",
    "Train":                            "[Train]",
    "Bus":                              "[Bus]",
    "Aircraft":                         "[Aircraft]",
    "Helicopter":                       "[Helicopter]",
    "Honk":                             "[Horn Honking]",
    "Car alarm":                        "[Car Alarm]",
    "Skidding":                         "[Tires Screeching]",
    "Engine starting":                  "[Engine Starting]",
    "Brake":                            "[Braking]",

    # ── MUSIC ────────────────────────────────────────────────────────────
    # Note: Most music events are suppressed in Phase 5.
    # Only music onsets reach here.
    "Music":                            "[Music]",
    "Music of Bollywood":               "[Music]",
    "Tabla music":                      "[Tabla Music]",
    "Classical music":                  "[Classical Music]",
    "Mantra":                           "[Chanting]",
    "Singing":                          "[Singing]",

    # ── DOMESTIC / ENVIRONMENT ───────────────────────────────────────────
    "Chopping (food)":                  "[Chopping Sounds]",
    "Mechanisms":                       "[Mechanical Sound]",
    "Ratchet, pawl":                    "[Clicking Mechanism]",
    "Pulleys":                          "[Pulley Sound]",
    "Typing":                           "[Typing]",
    "Door":                             "[Door]",
    "Squeak":                           "[Squeaking]",
    "Clock":                            "[Clock Ticking]",
    "Tick-tock":                        "[Clock Ticking]",

    # ── WEATHER / NATURE ─────────────────────────────────────────────────
    "Thunder":                          "[Thunder]",
    "Rain":                             "[Rain]",
    "Rain on surface":                  "[Rain]",
    "Wind":                             "[Wind]",
    "Fire":                             "[Fire]",
    "Water":                            "[Water]",
    "Splash, splatter":                 "[Splashing]",
}


# =============================================================================
# SECTION: Context-Specific Overrides
# =============================================================================
# Some labels get different CC text based on filter_reason or other context.
# Format: (audioset_label, filter_reason) → cc_text

CONTEXT_OVERRIDES = {
    # Music onset gets a clean [Music] regardless of sub-label
    ("Music", "music_onset"):           "[Music]",
    ("Music of Bollywood", "music_onset"): "[Music]",
    ("Tabla music", "music_onset"):     "[Tabla Music]",
    ("Mantra", "music_onset"):          "[Chanting]",
    ("Singing", "music_onset"):         "[Singing]",

    # Crowd in high-value context (narrative crowd reaction)
    ("Crowd", "high_value"):            "[Crowd Cheering]",

    # Screaming in high-value context
    ("Screaming", "high_value"):        "[Screaming]",
}


# =============================================================================
# SECTION: Fallback Label Generator
# =============================================================================

def generate_fallback_label(audioset_label: str) -> str:
    """
    Generate a CC label for any AudioSet label not in LABEL_MAP.

    Transformation rules (applied in order):
      1. Take text before the first comma (if present)
         "Ratchet, pawl" → "Ratchet"
      2. Remove parenthetical content
         "Chopping (food)" → "Chopping"
      3. Strip leading/trailing whitespace
      4. Title case: capitalize first letter of each word
         "gun shot" → "Gun Shot"
      5. Wrap in brackets
         "Ratchet" → "[Ratchet]"

    The result is always readable and in correct CC format,
    even if not perfectly descriptive.
    """
    # Step 1: Take text before first comma
    label = audioset_label.split(",")[0].strip()

    # Step 2: Remove parenthetical content — anything inside ()
    # re.sub replaces regex matches with empty string
    # \s* matches optional whitespace before the parenthesis
    label = re.sub(r"\s*\(.*?\)", "", label).strip()

    # Step 3: Handle edge cases
    if not label:
        # If everything was in parentheses, use original
        label = audioset_label.strip()

    # Step 4: Title case
    # str.title() capitalizes first letter of each word
    label = label.title()

    # Step 5: Wrap in brackets
    return f"[{label}]"


# =============================================================================
# SECTION: Core Label Generator
# =============================================================================

class LabelGenerator:
    """
    Generates human-readable CC text labels for accepted CCDecision objects.

    Populates the cc_label field on each CCDecision that has accepted=True.
    Rejected events also get a cc_label for debugging purposes.
    """

    def __init__(
        self,
        custom_label_map: Optional[dict] = None,
        custom_context_overrides: Optional[dict] = None,
    ):
        """
        Initialize the label generator.

        Args:
            custom_label_map: Additional label mappings to merge with
                              the default LABEL_MAP. Custom entries
                              override defaults. Loaded from config.yaml
                              in Phase 13.
            custom_context_overrides: Additional context-specific overrides
                              to merge with CONTEXT_OVERRIDES.
        """
        # Build effective label map
        self.label_map = LABEL_MAP.copy()
        if custom_label_map:
            self.label_map.update(custom_label_map)
            logger.info(
                f"Custom label map: {len(custom_label_map)} entries added"
            )

        # Build effective context overrides
        self.context_overrides = CONTEXT_OVERRIDES.copy()
        if custom_context_overrides:
            self.context_overrides.update(custom_context_overrides)

        logger.info(
            f"LabelGenerator initialized: "
            f"{len(self.label_map)} label mappings, "
            f"{len(self.context_overrides)} context overrides"
        )


    # ─────────────────────────────────────────────────────────────────────
    # SECTION: Single Label Generation
    # ─────────────────────────────────────────────────────────────────────

    def generate_label(self, decision) -> str:
        """
        Generate CC text for a single CCDecision.

        Lookup order:
          1. Context override (label + filter_reason combination)
          2. Primary label map (label only)
          3. Fallback generator (transformation rules)

        Args:
            decision: CCDecision from decision_engine.py

        Returns:
            CC text string e.g. "[Dog Barking]"
        """
        label = decision.label
        filter_reason = decision.filter_reason

        # Step 1: Check context-specific override
        context_key = (label, filter_reason)
        if context_key in self.context_overrides:
            cc_text = self.context_overrides[context_key]
            logger.debug(
                f"Context override: '{label}' + '{filter_reason}' "
                f"→ '{cc_text}'"
            )
            return cc_text

        # Step 2: Check primary label map
        if label in self.label_map:
            cc_text = self.label_map[label]
            logger.debug(f"Label map: '{label}' → '{cc_text}'")
            return cc_text

        # Step 3: Fallback generation
        cc_text = generate_fallback_label(label)
        logger.info(
            f"Fallback label generated: '{label}' → '{cc_text}' "
            f"(not in label map — consider adding to LABEL_MAP)"
        )
        return cc_text


    # ─────────────────────────────────────────────────────────────────────
    # SECTION: Batch Label Generation
    # ─────────────────────────────────────────────────────────────────────

    def generate_labels(self, decisions: list) -> list:
        """
        Generate CC labels for all CCDecision objects.

        Populates the cc_label field on each decision in-place.
        Returns the same list with cc_label fields populated.

        We label ALL decisions (accepted and rejected) for
        completeness — rejected events may be reviewed by editors
        and having their labels makes the review table readable.

        Args:
            decisions: list[CCDecision] from decision_engine.py

        Returns:
            Same list with cc_label populated on all items.
        """
        logger.info("=" * 60)
        logger.info(
            f"Generating CC labels for {len(decisions)} decisions"
        )
        logger.info("=" * 60)

        fallback_count = 0
        map_count = 0
        override_count = 0

        for decision in decisions:
            cc_label = self.generate_label(decision)
            decision.cc_label = cc_label

            # Track which path was used for each label
            context_key = (decision.label, decision.filter_reason)
            if context_key in self.context_overrides:
                override_count += 1
                source = "override"
            elif decision.label in self.label_map:
                map_count += 1
                source = "map"
            else:
                fallback_count += 1
                source = "fallback"

            status = "✓" if decision.accepted else "✗"
            logger.info(
                f"  {status} '{decision.label}' "
                f"→ '{cc_label}' [{source}]"
            )

        logger.info("=" * 60)
        logger.info(
            f"Label generation complete: "
            f"map={map_count}, "
            f"override={override_count}, "
            f"fallback={fallback_count}"
        )
        logger.info("=" * 60)

        return decisions


    # ─────────────────────────────────────────────────────────────────────
    # SECTION: Summary
    # ─────────────────────────────────────────────────────────────────────

    def summarize(self, decisions: list) -> str:
        """
        Display final CC annotations summary — accepted events only.
        This is the closest preview of what the SRT file will contain.
        """
        accepted = [d for d in decisions if d.accepted]

        if not accepted:
            return "No events accepted for CC annotation."

        lines = [
            f"{'─' * 60}",
            f"FINAL CC ANNOTATIONS ({len(accepted)} events)",
            f"{'─' * 60}",
            f"{'Time':>8}  {'CC Label':<30}  {'Source Label'}",
            f"{'─' * 60}",
        ]

        for d in accepted:
            mins = int(d.start_time // 60)
            secs = d.start_time % 60
            lines.append(
                f"{mins:02d}:{secs:05.2f}  "
                f"{d.cc_label:<30}  "
                f"({d.label})"
            )

        lines.append(f"{'─' * 60}")
        lines.append(
            f"These {len(accepted)} annotations will be written "
            f"to the SRT file."
        )
        return "\n".join(lines)