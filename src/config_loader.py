# =============================================================================
# Configuration Layer — YAML Config Loader
# =============================================================================
# Reads config.yaml and distributes validated parameters to all pipeline
# components. Every hardcoded value across Phases 2–12 is now sourced
# from this single file.
# =============================================================================

import yaml
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("config_loader")


# =============================================================================
# SECTION: Config Dataclasses
# One dataclass per pipeline phase — mirrors config.yaml structure exactly.
# =============================================================================

@dataclass
class PathsConfig:
    input_video: str
    extracted_dir: str
    output_srt: str
    panns_checkpoint: str
    panns_labels_csv: str
    face_landmarker_model: str
    pose_landmarker_model: str


@dataclass
class IngestionConfig:
    extraction_fps: float
    audio_sample_rate: int
    overwrite: bool


@dataclass
class AudioProcessorConfig:
    hop_duration: float
    normalize: bool
    normalization_method: str
    compute_spectrogram: bool


@dataclass
class SoundDetectorConfig:
    device: str
    batch_size: int
    top_k: int
    filter_speech: bool


@dataclass
class EventFilterConfig:
    high_value_threshold: float
    medium_value_threshold: float
    merge_gap_seconds: float
    suppress_sustained_music: bool
    music_onset_threshold: float
    music_silence_threshold: float
    min_music_silence_gap: float
    extra_suppress_labels: list
    extra_high_value_labels: list


@dataclass
class FrameExtractorConfig:
    pre_window_seconds: float
    post_window_seconds: float


@dataclass
class FaceAnalyzerConfig:
    max_faces: int
    min_detection_confidence: float
    ear_weight: float
    brow_weight: float
    mar_weight: float
    reaction_sensitivity: float


@dataclass
class PoseAnalyzerConfig:
    min_detection_confidence: float
    head_weight: float
    shoulder_weight: float
    lean_weight: float
    reaction_sensitivity: float


@dataclass
class VisualScorerConfig:
    face_weight: float
    pose_weight: float
    single_signal_discount: float
    duplicate_time_threshold: float


@dataclass
class DecisionEngineConfig:
    audio_weight: float
    visual_weight: float
    cc_threshold: float
    high_value_threshold_delta: float
    music_onset_threshold_delta: float
    medium_value_threshold_delta: float
    unknown_threshold_delta: float
    high_value_boost: float
    high_value_boost_min_audio: float
    neutral_visual_weight_reduction: float


@dataclass
class LabelGeneratorConfig:
    custom_labels: dict
    custom_context_overrides: dict


@dataclass
class SRTWriterConfig:
    min_display_duration: float
    max_display_duration: float
    encoding: str
    write_bom: bool


@dataclass
class PipelineConfig:
    """
    Master config object holding all phase configs.
    This is what gets passed around the pipeline.
    Access any phase's config via: config.decision_engine.cc_threshold
    """
    paths:           PathsConfig
    ingestion:       IngestionConfig
    audio_processor: AudioProcessorConfig
    sound_detector:  SoundDetectorConfig
    event_filter:    EventFilterConfig
    frame_extractor: FrameExtractorConfig
    face_analyzer:   FaceAnalyzerConfig
    pose_analyzer:   PoseAnalyzerConfig
    visual_scorer:   VisualScorerConfig
    decision_engine: DecisionEngineConfig
    label_generator: LabelGeneratorConfig
    srt_writer:      SRTWriterConfig


# =============================================================================
# SECTION: Config Loader
# =============================================================================

class ConfigLoader:
    """
    Loads, validates, and distributes pipeline configuration from config.yaml.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {self.config_path}\n"
                "Create config.yaml in the project root directory."
            )

    def load(self) -> PipelineConfig:
        """
        Load config.yaml, apply any active profile, validate, and return
        a PipelineConfig object.
        """
        logger.info(f"Loading configuration from: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        # Apply active profile if set
        raw = self._apply_profile(raw)

        # Build and validate each section
        config = PipelineConfig(
            paths=self._load_paths(raw["paths"]),
            ingestion=self._load_ingestion(raw["ingestion"]),
            audio_processor=self._load_audio_processor(raw["audio_processor"]),
            sound_detector=self._load_sound_detector(raw["sound_detector"]),
            event_filter=self._load_event_filter(raw["event_filter"]),
            frame_extractor=self._load_frame_extractor(raw["frame_extractor"]),
            face_analyzer=self._load_face_analyzer(raw["face_analyzer"]),
            pose_analyzer=self._load_pose_analyzer(raw["pose_analyzer"]),
            visual_scorer=self._load_visual_scorer(raw["visual_scorer"]),
            decision_engine=self._load_decision_engine(raw["decision_engine"]),
            label_generator=self._load_label_generator(raw["label_generator"]),
            srt_writer=self._load_srt_writer(raw["srt_writer"]),
        )

        self._validate(config)

        logger.info("Configuration loaded and validated successfully.")
        logger.info(
            f"Active profile: {raw.get('active_profile', 'none (using individual settings)')}"
        )
        return config


    # ─────────────────────────────────────────────────────────────────────
    # SECTION: Profile Application
    # ─────────────────────────────────────────────────────────────────────

    def _apply_profile(self, raw: dict) -> dict:
        """
        If active_profile is set, override individual settings with
        profile values.

        Profile values are flat (not nested by phase) — we apply them
        to the correct nested sections here.
        """
        active_profile = raw.get("active_profile")

        if not active_profile:
            return raw  # no profile — use individual settings

        profiles = raw.get("profiles", {})
        if active_profile not in profiles:
            raise ValueError(
                f"active_profile='{active_profile}' not found in profiles. "
                f"Available: {list(profiles.keys())}"
            )

        profile_values = profiles[active_profile]
        logger.info(
            f"Applying profile '{active_profile}': "
            f"{list(profile_values.keys())}"
        )

        # Map flat profile keys to their nested config sections
        profile_key_map = {
            "cc_threshold":          ("decision_engine", "cc_threshold"),
            "high_value_threshold":  ("event_filter",    "high_value_threshold"),
            "medium_value_threshold":("event_filter",    "medium_value_threshold"),
            "high_value_boost":      ("decision_engine", "high_value_boost"),
            "suppress_sustained_music": ("event_filter", "suppress_sustained_music"),
            "extra_suppress_labels": ("event_filter",    "extra_suppress_labels"),
        }

        for key, value in profile_values.items():
            if key in profile_key_map:
                section, param = profile_key_map[key]
                raw[section][param] = value
                logger.debug(f"  Profile override: {section}.{param} = {value}")
            else:
                logger.warning(f"  Unknown profile key: '{key}' — ignored")

        return raw


    # ─────────────────────────────────────────────────────────────────────
    # SECTION: Section Loaders
    # ─────────────────────────────────────────────────────────────────────

    def _load_paths(self, s: dict) -> PathsConfig:
        return PathsConfig(
            input_video=str(s["input_video"]),
            extracted_dir=str(s["extracted_dir"]),
            output_srt=str(s["output_srt"]),
            panns_checkpoint=str(s["panns_checkpoint"]),
            panns_labels_csv=str(s["panns_labels_csv"]),
            face_landmarker_model=str(s["face_landmarker_model"]),
            pose_landmarker_model=str(s["pose_landmarker_model"]),
        )

    def _load_ingestion(self, s: dict) -> IngestionConfig:
        return IngestionConfig(
            extraction_fps=float(s["extraction_fps"]),
            audio_sample_rate=int(s["audio_sample_rate"]),
            overwrite=bool(s["overwrite"]),
        )

    def _load_audio_processor(self, s: dict) -> AudioProcessorConfig:
        return AudioProcessorConfig(
            hop_duration=float(s["hop_duration"]),
            normalize=bool(s["normalize"]),
            normalization_method=str(s["normalization_method"]),
            compute_spectrogram=bool(s["compute_spectrogram"]),
        )

    def _load_sound_detector(self, s: dict) -> SoundDetectorConfig:
        return SoundDetectorConfig(
            device=str(s["device"]),
            batch_size=int(s["batch_size"]),
            top_k=int(s["top_k"]),
            filter_speech=bool(s["filter_speech"]),
        )

    def _load_event_filter(self, s: dict) -> EventFilterConfig:
        return EventFilterConfig(
            high_value_threshold=float(s["high_value_threshold"]),
            medium_value_threshold=float(s["medium_value_threshold"]),
            merge_gap_seconds=float(s["merge_gap_seconds"]),
            suppress_sustained_music=bool(s["suppress_sustained_music"]),
            music_onset_threshold=float(s["music_onset_threshold"]),
            music_silence_threshold=float(s["music_silence_threshold"]),
            min_music_silence_gap=float(s["min_music_silence_gap"]),
            extra_suppress_labels=list(s.get("extra_suppress_labels") or []),
            extra_high_value_labels=list(s.get("extra_high_value_labels") or []),
        )

    def _load_frame_extractor(self, s: dict) -> FrameExtractorConfig:
        return FrameExtractorConfig(
            pre_window_seconds=float(s["pre_window_seconds"]),
            post_window_seconds=float(s["post_window_seconds"]),
        )

    def _load_face_analyzer(self, s: dict) -> FaceAnalyzerConfig:
        return FaceAnalyzerConfig(
            max_faces=int(s["max_faces"]),
            min_detection_confidence=float(s["min_detection_confidence"]),
            ear_weight=float(s["ear_weight"]),
            brow_weight=float(s["brow_weight"]),
            mar_weight=float(s["mar_weight"]),
            reaction_sensitivity=float(s["reaction_sensitivity"]),
        )

    def _load_pose_analyzer(self, s: dict) -> PoseAnalyzerConfig:
        return PoseAnalyzerConfig(
            min_detection_confidence=float(s["min_detection_confidence"]),
            head_weight=float(s["head_weight"]),
            shoulder_weight=float(s["shoulder_weight"]),
            lean_weight=float(s["lean_weight"]),
            reaction_sensitivity=float(s["reaction_sensitivity"]),
        )

    def _load_visual_scorer(self, s: dict) -> VisualScorerConfig:
        return VisualScorerConfig(
            face_weight=float(s["face_weight"]),
            pose_weight=float(s["pose_weight"]),
            single_signal_discount=float(s["single_signal_discount"]),
            duplicate_time_threshold=float(s["duplicate_time_threshold"]),
        )

    def _load_decision_engine(self, s: dict) -> DecisionEngineConfig:
        return DecisionEngineConfig(
            audio_weight=float(s["audio_weight"]),
            visual_weight=float(s["visual_weight"]),
            cc_threshold=float(s["cc_threshold"]),
            high_value_threshold_delta=float(s["high_value_threshold_delta"]),
            music_onset_threshold_delta=float(s["music_onset_threshold_delta"]),
            medium_value_threshold_delta=float(s["medium_value_threshold_delta"]),
            unknown_threshold_delta=float(s["unknown_threshold_delta"]),
            high_value_boost=float(s["high_value_boost"]),
            high_value_boost_min_audio=float(s["high_value_boost_min_audio"]),
            neutral_visual_weight_reduction=float(s["neutral_visual_weight_reduction"]),
        )

    def _load_label_generator(self, s: dict) -> LabelGeneratorConfig:
        return LabelGeneratorConfig(
            custom_labels=dict(s.get("custom_labels") or {}),
            custom_context_overrides=dict(
                s.get("custom_context_overrides") or {}
            ),
        )

    def _load_srt_writer(self, s: dict) -> SRTWriterConfig:
        return SRTWriterConfig(
            min_display_duration=float(s["min_display_duration"]),
            max_display_duration=float(s["max_display_duration"]),
            encoding=str(s["encoding"]),
            write_bom=bool(s["write_bom"]),
        )


    # ─────────────────────────────────────────────────────────────────────
    # SECTION: Validation
    # ─────────────────────────────────────────────────────────────────────

    def _validate(self, config: PipelineConfig):
        """
        Validate config values are within acceptable ranges.
        Raises ValueError with a clear message if anything is wrong.
        """
        errors = []

        # Threshold validation
        for name, value in [
            ("decision_engine.cc_threshold",
             config.decision_engine.cc_threshold),
            ("event_filter.high_value_threshold",
             config.event_filter.high_value_threshold),
            ("event_filter.medium_value_threshold",
             config.event_filter.medium_value_threshold),
            ("face_analyzer.min_detection_confidence",
             config.face_analyzer.min_detection_confidence),
        ]:
            if not 0.0 <= value <= 1.0:
                errors.append(f"{name}={value} must be between 0.0 and 1.0")

        # Weight validation
        face_pose_sum = (
            config.visual_scorer.face_weight +
            config.visual_scorer.pose_weight
        )
        if not 0.9 <= face_pose_sum <= 1.1:
            errors.append(
                f"visual_scorer face_weight + pose_weight = {face_pose_sum:.2f}, "
                f"should be close to 1.0"
            )

        # FPS validation
        if config.ingestion.extraction_fps <= 0:
            errors.append(
                f"ingestion.extraction_fps must be > 0, "
                f"got {config.ingestion.extraction_fps}"
            )

        # Device validation
        if config.sound_detector.device not in ("cpu", "cuda"):
            errors.append(
                f"sound_detector.device must be 'cpu' or 'cuda', "
                f"got '{config.sound_detector.device}'"
            )

        # Normalization method validation
        if config.audio_processor.normalization_method not in ("peak", "rms"):
            errors.append(
                f"audio_processor.normalization_method must be 'peak' or 'rms', "
                f"got '{config.audio_processor.normalization_method}'"
            )

        if errors:
            raise ValueError(
                f"Configuration validation failed:\n" +
                "\n".join(f"  - {e}" for e in errors)
            )

        logger.info(
            f"Validation passed: "
            f"cc_threshold={config.decision_engine.cc_threshold}, "
            f"high_value_thresh={config.event_filter.high_value_threshold}, "
            f"extraction_fps={config.ingestion.extraction_fps}"
        )


    def print_summary(self, config: PipelineConfig):
        """Print a human-readable configuration summary."""
        print("=" * 60)
        print("PIPELINE CONFIGURATION SUMMARY")
        print("=" * 60)
        print(f"Input video:      {config.paths.input_video}")
        print(f"Output SRT:       {config.paths.output_srt}")
        print(f"Extraction FPS:   {config.ingestion.extraction_fps}")
        print(f"PANNs device:     {config.sound_detector.device}")
        print(f"CC threshold:     {config.decision_engine.cc_threshold}")
        print(f"High-val thresh:  {config.event_filter.high_value_threshold}")
        print(f"Med-val thresh:   {config.event_filter.medium_value_threshold}")
        print(f"Audio weight:     {config.decision_engine.audio_weight}")
        print(f"Visual weight:    {config.decision_engine.visual_weight}")
        print(f"Face weight:      {config.visual_scorer.face_weight}")
        print(f"Pose weight:      {config.visual_scorer.pose_weight}")
        print(f"Suppress music:   {config.event_filter.suppress_sustained_music}")
        print("=" * 60)