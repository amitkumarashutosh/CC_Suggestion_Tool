#!/usr/bin/env python3
# main.py
# =============================================================================
# CC Suggestion Tool — Main Pipeline Entry Point
# =============================================================================
# Runs the complete closed caption suggestion pipeline from video input
# to SRT output. All configuration is read from config.yaml.
#
# Usage:
#   python3 main.py
#   python3 main.py --video path/to/video.mp4
#   python3 main.py --video path/to/video.mp4 --output path/to/output.srt
#   python3 main.py --config path/to/config.yaml
#   python3 main.py --profile aggressive
#   python3 main.py --threshold 0.55
# =============================================================================

import argparse
import logging
import sys
import time
import os
from pathlib import Path


# =============================================================================
# SECTION: Logging Setup
# Set up logging before any imports so all modules log correctly.
# =============================================================================

def setup_logging(verbose: bool = False) -> None:
    """
    Configure logging for the full pipeline run.

    verbose=False: INFO level — shows phase progress and key decisions
    verbose=True:  DEBUG level — shows every frame analysis, every window
    """
    level = logging.DEBUG if verbose else logging.INFO

    # Clean format for INFO: timestamp + level + module + message
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Suppress MediaPipe C++ telemetry noise at WARNING level
    # (the "Failed to send to clearcut" messages)
    logging.getLogger("mediapipe").setLevel(logging.ERROR)


# =============================================================================
# SECTION: Argument Parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    All arguments are optional — defaults come from config.yaml.
    Command-line arguments OVERRIDE config.yaml values when provided.
    """
    parser = argparse.ArgumentParser(
        description=(
            "CC Suggestion Tool — Intelligent Closed Caption Generator\n"
            "Analyzes video audio and visual reactions to suggest CC annotations."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 main.py
      Run with all settings from config.yaml

  python3 main.py --video myvideo.mp4
      Override input video from config.yaml

  python3 main.py --video myvideo.mp4 --output myvideo.srt
      Override both input and output paths

  python3 main.py --profile aggressive
      Use the aggressive threshold profile (more CC annotations)

  python3 main.py --threshold 0.55
      Override cc_threshold to 0.55

  python3 main.py --video myvideo.mp4 --verbose
      Run with debug logging (shows every analysis step)
        """
    )

    parser.add_argument(
        "--video", "-v",
        type=str,
        default=None,
        help="Path to input video file. Overrides config.yaml paths.input_video"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path for output SRT file. Overrides config.yaml paths.output_srt"
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)"
    )

    parser.add_argument(
        "--profile", "-p",
        type=str,
        default=None,
        choices=["aggressive", "conservative", "music_video"],
        help="Override active_profile in config.yaml"
    )

    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=None,
        help="Override cc_threshold in config.yaml (0.0-1.0)"
    )

    parser.add_argument(
        "--skip-visual",
        action="store_true",
        default=False,
        help=(
            "Skip face and pose analysis (faster, audio-only decisions). "
            "Visual confidence will be 0.5 (neutral) for all events."
        )
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable debug logging (shows every analysis step)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Run full pipeline but do not write SRT file (preview only)"
    )

    return parser.parse_args()


# =============================================================================
# SECTION: Progress Reporter
# =============================================================================

class PipelineProgress:
    """
    Tracks and displays pipeline progress across all phases.
    Provides timing information per phase and overall.
    """

    def __init__(self, total_phases: int = 7):
        self.total_phases = total_phases
        self.current_phase = 0
        self.phase_times = {}
        self.pipeline_start = time.time()
        self._phase_start = None

    def start_phase(self, name: str) -> None:
        self.current_phase += 1
        self._phase_start = time.time()
        print(
            f"\n{'─' * 60}\n"
            f"[{self.current_phase}/{self.total_phases}] {name}\n"
            f"{'─' * 60}"
        )

    def end_phase(self, name: str, summary: str = "") -> None:
        elapsed = time.time() - self._phase_start
        self.phase_times[name] = elapsed
        status = f"  ✓ Done ({elapsed:.1f}s)"
        if summary:
            status += f" — {summary}"
        print(status)

    def pipeline_summary(self, decisions: list, output_path: str) -> None:
        total_time = time.time() - self.pipeline_start
        accepted = [d for d in decisions if d.accepted]

        print(f"\n{'═' * 60}")
        print(f"PIPELINE COMPLETE")
        print(f"{'═' * 60}")
        print(f"\nOutput SRT: {output_path}")
        print(f"\nCC Annotations Generated: {len(accepted)}")
        print(f"{'─' * 40}")

        for d in accepted:
            mins = int(d.start_time // 60)
            secs = d.start_time % 60
            print(
                f"  {mins:02d}:{secs:05.2f}  "
                f"{d.cc_label:<30}  "
                f"(score={d.cc_score:.3f})"
            )

        print(f"\n{'─' * 40}")
        print(f"Phase timing breakdown:")
        for phase_name, phase_time in self.phase_times.items():
            print(f"  {phase_name:<35} {phase_time:>5.1f}s")

        print(f"{'─' * 40}")
        print(f"  {'Total pipeline time':<35} {total_time:>5.1f}s")
        print(f"{'═' * 60}")
        print(
            f"\nTo preview: open your video in VLC, then"
            f"\n  Subtitle → Add Subtitle File → {output_path}"
        )


# =============================================================================
# SECTION: Main Pipeline
# =============================================================================

def run_pipeline(args: argparse.Namespace) -> int:
    """
    Run the complete CC suggestion pipeline.

    Returns:
        0 on success
        1 on error
    """

    # ─────────────────────────────────────────────────────────────────────
    # Import all pipeline modules
    # (imports here so logging is configured first)
    # ─────────────────────────────────────────────────────────────────────
    from src.config_loader import ConfigLoader
    from src.ingest import VideoIngestor
    from src.audio_processor import AudioProcessor
    from src.sound_detector import SoundDetector
    from src.event_filter import AudioEventFilter
    from src.frame_extractor import FrameExtractor
    from src.face_analyzer import FaceAnalyzer
    from src.pose_analyzer import PoseAnalyzer
    from src.visual_scorer import VisualScorer
    from src.decision_engine import CCDecisionEngine
    from src.label_generator import LabelGenerator
    from src.srt_writer import SRTWriter

    logger = logging.getLogger("main")

    # ─────────────────────────────────────────────────────────────────────
    # Print header
    # ─────────────────────────────────────────────────────────────────────
    print("═" * 60)
    print("  CC SUGGESTION TOOL — Intelligent Closed Caption Generator")
    print("  For Hindi and Regional-Language Video Content")
    print("═" * 60)

    # ─────────────────────────────────────────────────────────────────────
    # Phase 0: Load Configuration
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n[CONFIG] Loading: {args.config}")

    try:
        loader = ConfigLoader(args.config)

        # Apply command-line profile override before loading
        if args.profile:
            import yaml
            with open(args.config) as f:
                raw = yaml.safe_load(f)
            raw["active_profile"] = args.profile
            import tempfile
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.yaml', delete=False
            ) as tmp:
                yaml.dump(raw, tmp)
                tmp_config_path = tmp.name
            loader = ConfigLoader(tmp_config_path)

        config = loader.load()

    except Exception as e:
        print(f"\nERROR loading config: {e}")
        return 1

    # Apply command-line overrides on top of config
    if args.video:
        config.paths.input_video = args.video
    if args.output:
        config.paths.output_srt = args.output
    if args.threshold is not None:
        config.decision_engine.cc_threshold = args.threshold
        print(f"[CONFIG] cc_threshold overridden to {args.threshold}")

    # Validate input video exists
    if not Path(config.paths.input_video).exists():
        print(f"\nERROR: Input video not found: {config.paths.input_video}")
        return 1

    # Print configuration summary
    loader.print_summary(config)

    progress = PipelineProgress(total_phases=7)

    try:

        # ─────────────────────────────────────────────────────────────────
        # Phase 1: Video Ingestion
        # ─────────────────────────────────────────────────────────────────
        progress.start_phase("Video Ingestion (FFmpeg extraction)")

        ingestor = VideoIngestor(
            output_dir=config.paths.extracted_dir,
            target_sample_rate=config.ingestion.audio_sample_rate,
            target_fps=config.ingestion.extraction_fps,
            overwrite=config.ingestion.overwrite
        )
        extraction = ingestor.ingest(config.paths.input_video)

        progress.end_phase(
            "Video Ingestion",
            f"{extraction.num_frames_extracted} frames, "
            f"{extraction.duration_seconds:.0f}s video"
        )

        # ─────────────────────────────────────────────────────────────────
        # Phase 2: Audio Pre-Processing + Sound Event Detection
        # ─────────────────────────────────────────────────────────────────
        progress.start_phase(
            "Audio Analysis (pre-processing + PANNs inference)"
        )

        processor = AudioProcessor(
            hop_duration=config.audio_processor.hop_duration,
            normalize=config.audio_processor.normalize,
            normalization_method=config.audio_processor.normalization_method,
            compute_spectrogram=config.audio_processor.compute_spectrogram
        )
        processed_audio = processor.process(extraction.audio_path)

        detector = SoundDetector(
            checkpoint_path=config.paths.panns_checkpoint,
            labels_csv_path=config.paths.panns_labels_csv,
            device=config.sound_detector.device,
            batch_size=config.sound_detector.batch_size,
            top_k=config.sound_detector.top_k,
            filter_speech=config.sound_detector.filter_speech
        )
        detection_result = detector.detect(processed_audio)

        progress.end_phase(
            "Audio Analysis",
            f"{len(detection_result.events)} raw detections "
            f"across {len(processed_audio.windows)} windows"
        )

        # ─────────────────────────────────────────────────────────────────
        # Phase 3: Audio Event Filtering
        # ─────────────────────────────────────────────────────────────────
        progress.start_phase("Audio Event Filtering")

        event_filter = AudioEventFilter(
            high_value_threshold=config.event_filter.high_value_threshold,
            medium_value_threshold=config.event_filter.medium_value_threshold,
            merge_gap_seconds=config.event_filter.merge_gap_seconds,
            suppress_sustained_music=config.event_filter.suppress_sustained_music,
            music_onset_threshold=config.event_filter.music_onset_threshold,
            music_silence_threshold=config.event_filter.music_silence_threshold,
            min_music_silence_gap=config.event_filter.min_music_silence_gap,
            extra_suppress_labels=set(config.event_filter.extra_suppress_labels),
            extra_high_value_labels=set(config.event_filter.extra_high_value_labels)
        )
        filtered_events = event_filter.filter(detection_result)

        if not filtered_events:
            print(
                "\n  WARNING: No events passed audio filtering.\n"
                "  The video may have no detectable non-speech audio events,\n"
                "  or thresholds may be too high. Check config.yaml."
            )

        progress.end_phase(
            "Audio Event Filtering",
            f"{len(filtered_events)} candidate events "
            f"(from {len(detection_result.events)} raw)"
        )

        # ─────────────────────────────────────────────────────────────────
        # Phase 4: Visual Analysis
        # ─────────────────────────────────────────────────────────────────
        progress.start_phase("Visual Analysis (frame extraction + face + pose)")

        # Frame extraction
        extractor = FrameExtractor(
            frames_dir=str(
                Path(config.paths.extracted_dir) / "frames"
            ),
            extraction_fps=config.ingestion.extraction_fps,
            pre_window_seconds=config.frame_extractor.pre_window_seconds,
            post_window_seconds=config.frame_extractor.post_window_seconds
        )
        windows = extractor.extract_for_events(filtered_events)

        if args.skip_visual:
            # Audio-only mode: return neutral visual scores for all events
            logger.info("--skip-visual: skipping face and pose analysis")

            from src.face_analyzer import FaceReactionResult
            from src.pose_analyzer import PoseReactionResult

            face_results = [
                FaceReactionResult(
                    event_label=e.label,
                    event_timestamp=e.start_time,
                    face_reaction_score=0.5,
                    pre_features=[], post_features=[],
                    delta_ear=0.0, delta_brow=0.0, delta_mar=0.0,
                    faces_detected=False, detection_rate=0.0
                )
                for e in filtered_events
            ]
            pose_results = [
                PoseReactionResult(
                    event_label=e.label,
                    event_timestamp=e.start_time,
                    pose_reaction_score=0.5,
                    pre_snapshots=[], post_snapshots=[],
                    delta_head=0.0, delta_shoulder=0.0, delta_lean=0.0,
                    pose_detected=False, detection_rate=0.0
                )
                for e in filtered_events
            ]
            visual_note = "skipped (audio-only mode)"

        else:
            # Full visual analysis
            face_analyzer = FaceAnalyzer(
                max_faces=config.face_analyzer.max_faces,
                min_detection_confidence=config.face_analyzer.min_detection_confidence,
                ear_weight=config.face_analyzer.ear_weight,
                brow_weight=config.face_analyzer.brow_weight,
                mar_weight=config.face_analyzer.mar_weight,
                reaction_sensitivity=config.face_analyzer.reaction_sensitivity,
                model_path=config.paths.face_landmarker_model
            )
            face_results = face_analyzer.analyze_windows(windows)

            pose_analyzer = PoseAnalyzer(
                model_path=config.paths.pose_landmarker_model,
                min_detection_confidence=config.pose_analyzer.min_detection_confidence,
                head_weight=config.pose_analyzer.head_weight,
                shoulder_weight=config.pose_analyzer.shoulder_weight,
                lean_weight=config.pose_analyzer.lean_weight,
                reaction_sensitivity=config.pose_analyzer.reaction_sensitivity
            )
            pose_results = pose_analyzer.analyze_windows(windows)

            faces_found = sum(1 for r in face_results if r.faces_detected)
            poses_found = sum(1 for r in pose_results if r.pose_detected)
            visual_note = (
                f"faces detected: {faces_found}/{len(face_results)}, "
                f"poses: {poses_found}/{len(pose_results)}"
            )

        progress.end_phase("Visual Analysis", visual_note)

        # ─────────────────────────────────────────────────────────────────
        # Phase 5: Visual Scoring + CC Decisions
        # ─────────────────────────────────────────────────────────────────
        progress.start_phase("CC Decision Engine")

        scorer = VisualScorer(
            face_weight=config.visual_scorer.face_weight,
            pose_weight=config.visual_scorer.pose_weight,
            single_signal_discount=config.visual_scorer.single_signal_discount,
            duplicate_time_threshold=config.visual_scorer.duplicate_time_threshold
        )
        scored_events = scorer.score(filtered_events, face_results, pose_results)

        engine = CCDecisionEngine(
            audio_weight=config.decision_engine.audio_weight,
            visual_weight=config.decision_engine.visual_weight,
            cc_threshold=config.decision_engine.cc_threshold,
            high_value_threshold_delta=config.decision_engine.high_value_threshold_delta,
            music_onset_threshold_delta=config.decision_engine.music_onset_threshold_delta,
            medium_value_threshold_delta=config.decision_engine.medium_value_threshold_delta,
            unknown_threshold_delta=config.decision_engine.unknown_threshold_delta,
            high_value_boost=config.decision_engine.high_value_boost,
            high_value_boost_min_audio=config.decision_engine.high_value_boost_min_audio,
            neutral_visual_weight_reduction=config.decision_engine.neutral_visual_weight_reduction
        )
        decisions = engine.decide(scored_events)

        accepted = [d for d in decisions if d.accepted]
        rejected = [d for d in decisions if not d.accepted]

        progress.end_phase(
            "CC Decision Engine",
            f"{len(accepted)} accepted, {len(rejected)} rejected"
        )

        if not accepted:
            print(
                "\n  WARNING: No events accepted for CC annotation.\n"
                "  Consider lowering cc_threshold in config.yaml\n"
                f"  (currently {config.decision_engine.cc_threshold}).\n"
                f"  Try: python3 main.py --threshold 0.50"
            )

        # ─────────────────────────────────────────────────────────────────
        # Phase 6: Label Generation
        # ─────────────────────────────────────────────────────────────────
        progress.start_phase("CC Label Generation")

        generator = LabelGenerator(
            custom_label_map=config.label_generator.custom_labels or None,
        )
        decisions = generator.generate_labels(decisions)

        progress.end_phase(
            "CC Label Generation",
            f"{len(accepted)} labels generated"
        )

        # ─────────────────────────────────────────────────────────────────
        # Phase 7: SRT File Generation
        # ─────────────────────────────────────────────────────────────────
        progress.start_phase("SRT File Generation")

        writer = SRTWriter(
            output_path=config.paths.output_srt,
            min_display_duration=config.srt_writer.min_display_duration,
            max_display_duration=config.srt_writer.max_display_duration,
            encoding=config.srt_writer.encoding,
            write_bom=config.srt_writer.write_bom
        )
        blocks = writer.build_blocks(decisions)

        if args.dry_run:
            print("\n  DRY RUN — SRT content preview:")
            print("  " + "─" * 40)
            for line in writer.preview(blocks).split("\n"):
                print(f"  {line}")
            print("  " + "─" * 40)
            print("  (SRT file not written — dry run mode)")
            output_path = "[dry run — no file written]"
        else:
            output_path = writer.write(blocks)
            validation = writer.validate(output_path)
            if not validation["valid"]:
                logger.error(
                    f"SRT validation failed: {validation['errors']}"
                )
            else:
                logger.info("SRT validation passed.")

        progress.end_phase(
            "SRT File Generation",
            f"{len(blocks)} blocks → {output_path}"
        )

        # ─────────────────────────────────────────────────────────────────
        # Final Summary
        # ─────────────────────────────────────────────────────────────────
        progress.pipeline_summary(decisions, output_path)

        # Clean up temp config if profile was applied
        if args.profile and 'tmp_config_path' in locals():
            try:
                os.unlink(tmp_config_path)
            except Exception:
                pass

        return 0

    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        return 1

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        print(f"\nERROR: {e}")
        print("Run with --verbose for full traceback.")
        return 1


# =============================================================================
# SECTION: Entry Point
# =============================================================================

def main():
    args = parse_args()
    setup_logging(verbose=args.verbose)
    sys.exit(run_pipeline(args))


if __name__ == "__main__":
    main()