# =============================================================================
# STAGE 5: SRT Output File Generator
# =============================================================================
# Takes accepted CCDecision objects from decision_engine.py (with cc_label
# populated by label_generator.py) and writes a valid .srt file to disk.
#
# Output format complies with the SRT specification:
#   https://docs.fileformat.com/video/srt/
#
# Compatible with: VLC, YouTube Studio, Adobe Premiere, DaVinci Resolve,
#                  Windows Media Player, QuickTime, FFmpeg subtitle tracks.
# =============================================================================

import os
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("srt_writer")


# =============================================================================
# SECTION: Timestamp Conversion
# =============================================================================

def seconds_to_srt_timestamp(total_seconds: float) -> str:
    """
    Convert a float number of seconds to SRT timestamp format.

    SRT format: HH:MM:SS,mmm
    Note: millisecond separator is a COMMA, not a period.
    This is the most common source of SRT compatibility errors.

    Args:
        total_seconds: Time in seconds (float). Can be 0.0 or any positive.

    Returns:
        String in format "HH:MM:SS,mmm"

    Examples:
        6.0    → "00:00:06,000"
        23.5   → "00:00:23,500"
        143.75 → "00:02:23,750"
        3661.1 → "01:01:01,100"
    """
    # Clamp to non-negative (defensive programming)
    total_seconds = max(0.0, float(total_seconds))

    # Extract components using integer division and modulo
    hours   = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)

    # Milliseconds: take the fractional part of total_seconds,
    # multiply by 1000, round to nearest integer
    # round() is important: 6.9995 → 7000ms not 6999ms
    millis  = int(round((total_seconds % 1) * 1000))

    # Handle millisecond overflow (rare but possible with floating point)
    # e.g. round(0.9995 * 1000) = 1000 → carry into seconds
    if millis >= 1000:
        millis -= 1000
        seconds += 1
    if seconds >= 60:
        seconds -= 60
        minutes += 1
    if minutes >= 60:
        minutes -= 60
        hours += 1

    # Format with zero-padding
    # :02d = at least 2 digits, zero-padded (hours, minutes, seconds)
    # :03d = at least 3 digits, zero-padded (milliseconds)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


# =============================================================================
# SECTION: Data Container for SRT Block
# =============================================================================

@dataclass
class SRTBlock:
    """
    A single block in an SRT file.

    Fields:
        sequence:    Block number (1-based, sequential)
        start_ts:    Start timestamp string "HH:MM:SS,mmm"
        end_ts:      End timestamp string "HH:MM:SS,mmm"
        text:        Caption text (e.g. "[Dog Barking]")
        source_label: Original AudioSet label (for metadata/debugging)
        cc_score:    The cc_score that led to this block's acceptance
    """
    sequence: int
    start_ts: str
    end_ts: str
    text: str
    source_label: str
    cc_score: float

    def to_srt_string(self) -> str:
        """
        Format this block as a valid SRT block string.

        Returns exactly:
            {sequence}\\n
            {start_ts} --> {end_ts}\\n
            {text}\\n
            \\n

        The trailing blank line is required by the SRT spec.
        It separates this block from the next block.
        """
        return (
            f"{self.sequence}\n"
            f"{self.start_ts} --> {self.end_ts}\n"
            f"{self.text}\n"
            f"\n"
        )

    def __repr__(self):
        return (
            f"SRTBlock({self.sequence}: "
            f"{self.start_ts} → {self.end_ts} | {self.text!r})"
        )


# =============================================================================
# SECTION: Core SRT Writer
# =============================================================================

class SRTWriter:
    """
    Converts accepted CCDecision objects into a valid SRT file.

    Usage:
        writer = SRTWriter(output_path="data/output/output.srt")
        srt_blocks = writer.build_blocks(decisions)
        writer.write(srt_blocks)
    """

    def __init__(
        self,
        output_path: str = "data/output/output.srt",
        min_display_duration: float = 2.0,
        max_display_duration: float = 5.0,
        encoding: str = "utf-8",
        write_bom: bool = False,
    ):
        """
        Initialize the SRT writer.

        Args:
            output_path:          Path where the .srt file will be written.
            min_display_duration: Minimum time a CC label stays on screen.
                                  2.0s = enough time to read any 1-4 word label.
                                  DCMP standard recommends 1.5s minimum.
            max_display_duration: Maximum time a CC label stays on screen.
                                  5.0s prevents labels from lingering too long
                                  after the sound has ended.
            encoding:             File encoding. Always "utf-8" for Unicode.
                                  Supports Hindi, Tamil, Arabic if needed.
            write_bom:            Write UTF-8 BOM (Byte Order Mark) at file start.
                                  Some Windows tools require this for correct
                                  Unicode detection. Default False — most modern
                                  tools handle UTF-8 without BOM.
        """
        self.output_path = Path(output_path)
        self.min_display_duration = min_display_duration
        self.max_display_duration = max_display_duration
        self.encoding = encoding
        self.write_bom = write_bom

        # Create output directory if it doesn't exist
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"SRTWriter initialized: "
            f"output={self.output_path}, "
            f"duration=[{min_display_duration}s, {max_display_duration}s], "
            f"encoding={encoding}"
        )


    # ─────────────────────────────────────────────────────────────────────
    # SECTION: Display Duration Calculation
    # ─────────────────────────────────────────────────────────────────────

    def _calculate_display_duration(self, decision) -> float:
        """
        Calculate how long the CC label should be displayed.

        Uses the detected event duration, clamped to [min, max].

        Args:
            decision: CCDecision with start_time and end_time

        Returns:
            Display duration in seconds (float)
        """
        raw_duration = decision.end_time - decision.start_time

        # Clamp to configured range
        display_duration = max(
            self.min_display_duration,
            min(raw_duration, self.max_display_duration)
        )

        return display_duration


    # ─────────────────────────────────────────────────────────────────────
    # SECTION: Block Building
    # ─────────────────────────────────────────────────────────────────────

    def build_blocks(self, decisions: list) -> list:
        """
        Convert accepted CCDecision objects into SRTBlock objects.

        Only processes decisions where accepted=True.
        Sorts by start_time before assigning sequence numbers.
        Handles overlapping events by adjusting end times.

        Args:
            decisions: list[CCDecision] from decision_engine.py
                       (with cc_label populated by label_generator.py)

        Returns:
            list[SRTBlock] sorted by start_time, numbered sequentially.
        """
        # Filter to accepted decisions only
        accepted = [d for d in decisions if d.accepted]

        if not accepted:
            logger.warning("No accepted decisions — SRT file will be empty.")
            return []

        # Verify cc_label is populated on all accepted decisions
        for d in accepted:
            if d.cc_label is None:
                raise ValueError(
                    f"cc_label is None for accepted decision '{d.label}'. "
                    "Run LabelGenerator.generate_labels() before SRTWriter."
                )

        # Sort by start_time (should already be sorted, but safety measure)
        accepted_sorted = sorted(accepted, key=lambda d: d.start_time)

        blocks = []
        for sequence, decision in enumerate(accepted_sorted, start=1):
            # Calculate display duration
            display_duration = self._calculate_display_duration(decision)

            # Calculate end timestamp for display
            # Note: this may differ from decision.end_time if clamped
            display_end = decision.start_time + display_duration

            # Convert to SRT timestamp strings
            start_ts = seconds_to_srt_timestamp(decision.start_time)
            end_ts   = seconds_to_srt_timestamp(display_end)

            # Sanity check: end must be after start
            if display_end <= decision.start_time:
                logger.warning(
                    f"Block {sequence}: end_time <= start_time for "
                    f"'{decision.label}'. Forcing 2s display duration."
                )
                display_end = decision.start_time + 2.0
                end_ts = seconds_to_srt_timestamp(display_end)

            block = SRTBlock(
                sequence=sequence,
                start_ts=start_ts,
                end_ts=end_ts,
                text=decision.cc_label,
                source_label=decision.label,
                cc_score=decision.cc_score
            )
            blocks.append(block)

            logger.debug(
                f"Block {sequence}: {start_ts} → {end_ts} | "
                f"'{decision.cc_label}' (from '{decision.label}', "
                f"duration={display_duration:.1f}s)"
            )

        logger.info(
            f"Built {len(blocks)} SRT blocks from "
            f"{len(accepted)} accepted decisions"
        )

        return blocks


    # ─────────────────────────────────────────────────────────────────────
    # SECTION: Overlap Detection and Resolution
    # ─────────────────────────────────────────────────────────────────────

    def _resolve_overlaps(self, blocks: list) -> list:
        """
        Detect and resolve overlapping SRT blocks.

        Two blocks overlap if the first block's display end time is
        after the second block's start time. Overlapping CC labels
        are confusing — viewers see two labels simultaneously.

        Resolution strategy: trim the end time of the earlier block
        to 0.1s before the start of the next block.

        This matters for events close in time (e.g. Music @ 23s
        and Bow-wow @ 25s — if Music displays for 5s, it would
        overlap with Bow-wow).

        Args:
            blocks: list[SRTBlock] sorted by start_time

        Returns:
            Same list with overlapping end times resolved.
        """
        if len(blocks) <= 1:
            return blocks

        for i in range(len(blocks) - 1):
            current = blocks[i]
            next_block = blocks[i + 1]

            # Parse current end time back to seconds for comparison
            # We need to re-parse because timestamps are strings
            current_end_seconds = self._ts_to_seconds(current.end_ts)
            next_start_seconds  = self._ts_to_seconds(next_block.start_ts)

            if current_end_seconds > next_start_seconds:
                # Overlap detected — trim current block's end
                # Set end to 0.1s before next block starts
                new_end_seconds = next_start_seconds - 0.1
                new_end_seconds = max(
                    new_end_seconds,
                    self._ts_to_seconds(current.start_ts) + self.min_display_duration
                )
                old_end = current.end_ts
                current.end_ts = seconds_to_srt_timestamp(new_end_seconds)
                logger.info(
                    f"Overlap resolved: block {current.sequence} "
                    f"'{current.text}' end trimmed "
                    f"{old_end} → {current.end_ts}"
                )

        return blocks


    def _ts_to_seconds(self, ts: str) -> float:
        """
        Parse SRT timestamp string back to seconds (float).

        Inverse of seconds_to_srt_timestamp().
        "00:02:23,750" → 143.75

        Used only internally for overlap detection.
        """
        # Split "HH:MM:SS,mmm" into components
        time_part, ms_part = ts.split(",")
        h, m, s = time_part.split(":")
        return (
            int(h) * 3600 +
            int(m) * 60  +
            int(s)       +
            int(ms_part) / 1000.0
        )


    # ─────────────────────────────────────────────────────────────────────
    # SECTION: File Writing
    # ─────────────────────────────────────────────────────────────────────

    def write(self, blocks: list) -> str:
        """
        Write SRT blocks to the output file.

        Args:
            blocks: list[SRTBlock] from build_blocks()

        Returns:
            Path to the written SRT file as a string.

        File structure:
            [block 1 content]\\n\\n
            [block 2 content]\\n\\n
            ...
            [last block content]\\n\\n

        Each block's to_srt_string() already includes the trailing \\n\\n.
        We just concatenate them.
        """
        if not blocks:
            logger.warning("No SRT blocks to write. Creating empty SRT file.")
            self.output_path.write_text("", encoding=self.encoding)
            return str(self.output_path)

        # Resolve any overlapping timestamps before writing
        blocks = self._resolve_overlaps(blocks)

        # Build the full SRT content string
        srt_content = ""
        for block in blocks:
            srt_content += block.to_srt_string()

        # Write to file
        # "utf-8-sig" encoding adds the UTF-8 BOM if write_bom=True
        # "utf-8" writes clean UTF-8 without BOM
        file_encoding = "utf-8-sig" if self.write_bom else "utf-8"

        with open(self.output_path, "w", encoding=file_encoding) as f:
            f.write(srt_content)

        file_size = self.output_path.stat().st_size
        logger.info(
            f"SRT file written: {self.output_path} "
            f"({file_size} bytes, {len(blocks)} blocks)"
        )

        return str(self.output_path)


    # ─────────────────────────────────────────────────────────────────────
    # SECTION: Validation
    # ─────────────────────────────────────────────────────────────────────

    def validate(self, srt_path: str) -> dict:
        """
        Validate a written SRT file for format compliance.

        Checks:
          - File exists and is non-empty
          - All sequence numbers are sequential starting from 1
          - All timestamps are in correct HH:MM:SS,mmm format
          - All end times are after start times
          - All blocks are separated by blank lines
          - No overlapping timestamp ranges

        Returns:
            dict with keys:
                valid: bool — True if file passes all checks
                errors: list of error strings (empty if valid)
                block_count: number of blocks found
        """
        errors = []
        block_count = 0

        if not Path(srt_path).exists():
            return {"valid": False, "errors": ["File not found"], "block_count": 0}

        content = Path(srt_path).read_text(encoding="utf-8-sig")

        if not content.strip():
            return {
                "valid": True,
                "errors": [],
                "block_count": 0,
                "note": "Empty SRT file (no events accepted)"
            }

        # Split into blocks on double newline
        # Strip to remove leading/trailing whitespace
        raw_blocks = [b.strip() for b in content.strip().split("\n\n")]
        raw_blocks = [b for b in raw_blocks if b]  # remove empty strings

        import re
        ts_pattern = re.compile(
            r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})"
        )

        prev_end_seconds = -1.0

        for i, raw_block in enumerate(raw_blocks):
            lines = raw_block.strip().split("\n")

            if len(lines) < 3:
                errors.append(
                    f"Block {i+1}: Too few lines ({len(lines)}, need at least 3)"
                )
                continue

            # Check sequence number
            try:
                seq = int(lines[0].strip())
                if seq != i + 1:
                    errors.append(
                        f"Block {i+1}: Sequence number {seq} "
                        f"(expected {i+1})"
                    )
            except ValueError:
                errors.append(
                    f"Block {i+1}: First line '{lines[0]}' is not a number"
                )

            # Check timestamp line
            ts_match = ts_pattern.match(lines[1].strip())
            if not ts_match:
                errors.append(
                    f"Block {i+1}: Invalid timestamp line: '{lines[1]}'"
                )
                continue

            start_ts, end_ts = ts_match.group(1), ts_match.group(2)
            start_s = self._ts_to_seconds(start_ts)
            end_s   = self._ts_to_seconds(end_ts)

            # Check end > start
            if end_s <= start_s:
                errors.append(
                    f"Block {i+1}: End time {end_ts} not after "
                    f"start time {start_ts}"
                )

            # Check no overlap with previous block
            if start_s < prev_end_seconds - 0.001:
                errors.append(
                    f"Block {i+1}: Start time {start_ts} overlaps "
                    f"with previous block end"
                )

            prev_end_seconds = end_s
            block_count += 1

            # Check caption text present
            if not lines[2].strip():
                errors.append(f"Block {i+1}: Empty caption text")

        result = {
            "valid": len(errors) == 0,
            "errors": errors,
            "block_count": block_count
        }
        return result


    # ─────────────────────────────────────────────────────────────────────
    # SECTION: Preview
    # ─────────────────────────────────────────────────────────────────────

    def preview(self, blocks: list) -> str:
        """
        Return the SRT content as a string for inspection before writing.
        Useful for logging and testing.
        """
        if not blocks:
            return "(no SRT blocks)"
        return "".join(block.to_srt_string() for block in blocks)