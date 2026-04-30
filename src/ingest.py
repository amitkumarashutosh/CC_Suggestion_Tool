# =============================================================================
# STAGE 1: Video Ingestion and Track Extraction
# =============================================================================
# This module accepts any video file and extracts:
#   1. A clean mono WAV audio file at 16kHz
#   2. A directory of JPEG frames at a configurable FPS
# It uses FFmpeg via Python's subprocess module.
# All downstream pipeline stages read from the outputs this module produces.
# =============================================================================

# SECTION: Imports
import subprocess   # runs FFmpeg as a child process
import os           # file/folder path operations
import logging      # structured log messages (better than bare print())
import json         # parses FFmpeg's JSON probe output
from pathlib import Path  # modern, readable path handling
from dataclasses import dataclass  # clean data containers
from typing import Optional  # type hints for optional values

# SECTION: Logging Configuration
# We configure logging here so every message has a timestamp and level.
# Levels: DEBUG < INFO < WARNING < ERROR < CRITICAL
# We use INFO for normal operation, ERROR for failures.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("ingest")


# SECTION: Data Container for Extraction Results
@dataclass
class ExtractionResult:
    """
    Holds the results of a successful video extraction.
    A dataclass automatically generates __init__, __repr__, and __eq__
    from the field definitions — no boilerplate needed.
    """
    audio_path: str        # absolute path to extracted audio.wav
    frames_dir: str        # absolute path to the frames/ directory
    duration_seconds: float  # total video duration in seconds
    fps: float             # frame rate of the original video
    sample_rate: int       # audio sample rate we extracted at (16000)
    num_frames_extracted: int  # how many frame images were saved


# SECTION: Core Ingestion Class
class VideoIngestor:
    """
    Handles all video ingestion operations.
    
    Design decision: We use a class (not standalone functions) so that
    configuration (output directories, target sample rate, target FPS)
    is set once at construction and reused across multiple calls.
    This matters when processing a batch of videos.
    """

    def __init__(
        self,
        output_dir: str = "data/extracted",
        target_sample_rate: int = 16000,
        target_fps: float = 1.0,
        overwrite: bool = False
    ):
        """
        Initialize the ingestor with output configuration.

        Args:
            output_dir: Where to save extracted audio and frames.
            target_sample_rate: Audio sample rate in Hz. Default 16000 (16kHz).
            target_fps: Frames to extract per second of video. Default 1.0.
                        Set to higher (e.g. 5.0) for finer visual analysis.
                        Set to lower (e.g. 0.5) to save disk space.
            overwrite: If True, re-extract even if output files exist.
        """
        self.output_dir = Path(output_dir)
        self.target_sample_rate = target_sample_rate
        self.target_fps = target_fps
        self.overwrite = overwrite

        # Create output directories if they don't exist.
        # exist_ok=True means: don't raise an error if directory already exists.
        self.audio_output_dir = self.output_dir
        self.frames_output_dir = self.output_dir / "frames"

        self.audio_output_dir.mkdir(parents=True, exist_ok=True)
        self.frames_output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"VideoIngestor initialized. Output dir: {self.output_dir}")
        logger.info(
            f"Config: sample_rate={target_sample_rate}Hz, "
            f"fps={target_fps}, overwrite={overwrite}"
        )


    # ─────────────────────────────────────────────────────────────────────
    # SECTION: Video Probing (read metadata before extraction)
    # ─────────────────────────────────────────────────────────────────────

    def probe_video(self, video_path: str) -> dict:
        """
        Use ffprobe (ships with FFmpeg) to read video metadata without
        decoding any frames or audio. Think of ffprobe as reading the
        table of contents of the video container.

        Returns a dict with keys: duration, fps, width, height,
        has_audio, has_video, audio_sample_rate.

        Why probe first?
        - We need to know the video's native FPS to validate our config
        - We need to confirm audio exists before trying to extract it
        - We need duration to estimate processing time and output size
        """
        video_path = str(video_path)

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        logger.info(f"Probing video: {video_path}")

        # ffprobe command:
        # -v quiet          → suppress FFmpeg's banner/log output
        # -print_format json → output metadata as JSON (easy to parse)
        # -show_streams     → show info about each stream (video + audio)
        # -show_format      → show container-level info (duration, size)
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            "-show_format",
            video_path
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"ffprobe failed on {video_path}.\n"
                f"Error: {result.stderr}"
            )

        # Parse the JSON output ffprobe gave us
        probe_data = json.loads(result.stdout)

        # Extract metadata from the parsed JSON
        metadata = self._parse_probe_output(probe_data)
        logger.info(
            f"Video probed: duration={metadata['duration']:.1f}s, "
            f"fps={metadata['fps']:.2f}, "
            f"has_audio={metadata['has_audio']}, "
            f"native_audio_rate={metadata['audio_sample_rate']}Hz"
        )
        return metadata


    def _parse_probe_output(self, probe_data: dict) -> dict:
        """
        Parse the raw ffprobe JSON output into a clean metadata dict.
        This is an internal helper — the underscore prefix is a Python
        convention meaning "not for external use."
        """
        metadata = {
            "duration": 0.0,
            "fps": 0.0,
            "width": 0,
            "height": 0,
            "has_video": False,
            "has_audio": False,
            "audio_sample_rate": 0,
            "video_codec": "unknown",
            "audio_codec": "unknown"
        }

        # Process each stream (video stream and audio stream are separate)
        for stream in probe_data.get("streams", []):
            codec_type = stream.get("codec_type", "")

            if codec_type == "video":
                metadata["has_video"] = True
                metadata["video_codec"] = stream.get("codec_name", "unknown")
                metadata["width"] = stream.get("width", 0)
                metadata["height"] = stream.get("height", 0)

                # FPS is stored as a fraction string like "30000/1001" (29.97fps)
                # or "25/1" (25fps). We evaluate it.
                fps_str = stream.get("r_frame_rate", "0/1")
                try:
                    numerator, denominator = fps_str.split("/")
                    metadata["fps"] = float(numerator) / float(denominator)
                except (ValueError, ZeroDivisionError):
                    metadata["fps"] = 0.0

            elif codec_type == "audio":
                metadata["has_audio"] = True
                metadata["audio_codec"] = stream.get("codec_name", "unknown")
                metadata["audio_sample_rate"] = int(
                    stream.get("sample_rate", 0)
                )

        # Duration is at the container (format) level, not stream level
        format_info = probe_data.get("format", {})
        try:
            metadata["duration"] = float(format_info.get("duration", 0.0))
        except (ValueError, TypeError):
            metadata["duration"] = 0.0

        return metadata


    # ─────────────────────────────────────────────────────────────────────
    # SECTION: Audio Extraction
    # ─────────────────────────────────────────────────────────────────────

    def extract_audio(self, video_path: str) -> str:
        """
        Extract the audio track from the video file and save as a
        16kHz mono WAV file.

        Returns the path to the extracted audio.wav file.

        Why WAV?
        WAV is uncompressed audio. There is no codec to decode — the file
        contains raw PCM (Pulse Code Modulation) samples. librosa and
        PyTorch both read WAV files natively and efficiently.
        """
        audio_output_path = self.audio_output_dir / "audio.wav"

        # Skip extraction if file already exists and overwrite is False.
        # This saves significant time when re-running the pipeline on the
        # same video (common during development and debugging).
        if audio_output_path.exists() and not self.overwrite:
            logger.info(
                f"Audio already extracted: {audio_output_path}. "
                "Skipping. (Set overwrite=True to re-extract.)"
            )
            return str(audio_output_path)

        logger.info(f"Extracting audio from: {video_path}")

        # FFmpeg command breakdown:
        # -i {video_path}     → input file (i = input)
        # -vn                 → no video (disable video stream output)
        # -acodec pcm_s16le   → audio codec: PCM signed 16-bit little-endian
        #                       This is standard uncompressed WAV format
        # -ac 1               → audio channels: 1 (mono)
        #                       ac = audio channels
        # -ar {sample_rate}   → audio rate (sample rate) in Hz
        # -y                  → overwrite output file without asking
        # {output_path}       → output file path (WAV extension triggers WAV muxer)
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vn",                          # no video
            "-acodec", "pcm_s16le",         # uncompressed PCM WAV
            "-ac", "1",                      # mono
            "-ar", str(self.target_sample_rate),  # 16kHz
            "-y",                            # overwrite without asking
            str(audio_output_path)
        ]

        logger.info(f"Running FFmpeg audio extraction...")
        logger.debug(f"Command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            # FFmpeg writes its logs to stderr even on success.
            # A non-zero return code means something actually failed.
            raise RuntimeError(
                f"FFmpeg audio extraction failed.\n"
                f"Command: {' '.join(cmd)}\n"
                f"Stderr: {result.stderr[-2000:]}"  # last 2000 chars of error
            )

        # Verify the output file actually exists and is non-empty
        if not audio_output_path.exists():
            raise RuntimeError(
                f"FFmpeg reported success but output file not found: "
                f"{audio_output_path}"
            )

        file_size_mb = audio_output_path.stat().st_size / (1024 * 1024)
        logger.info(
            f"Audio extracted: {audio_output_path} "
            f"({file_size_mb:.2f} MB)"
        )

        return str(audio_output_path)


    # ─────────────────────────────────────────────────────────────────────
    # SECTION: Frame Extraction
    # ─────────────────────────────────────────────────────────────────────

    def extract_frames(self, video_path: str) -> str:
        """
        Extract video frames as JPEG images at the configured FPS.

        Each frame is saved as: frames/frame_XXXXXX.jpg
        where XXXXXX is the frame number padded to 6 digits.

        Example at 1 FPS for a 120-second video:
            frames/frame_000001.jpg  ← second 1
            frames/frame_000002.jpg  ← second 2
            ...
            frames/frame_000120.jpg  ← second 120

        Returns the path to the frames directory.

        Why JPEG and not PNG?
        JPEG is lossy compression. PNG is lossless. For visual reaction
        detection (detecting surprise, head turns), JPEG quality 95 is
        indistinguishable from PNG and produces files 5–10x smaller.
        For 120 frames at 1920×1080: PNG≈60MB, JPEG≈8MB.
        """
        frames_dir = self.frames_output_dir

        # Check if frames already exist (skip if so, same as audio)
        existing_frames = list(frames_dir.glob("frame_*.jpg"))
        if existing_frames and not self.overwrite:
            logger.info(
                f"Frames already extracted: {len(existing_frames)} frames "
                f"in {frames_dir}. Skipping."
            )
            return str(frames_dir)

        # Clean the frames directory before extracting to avoid mixing
        # frames from different videos
        if frames_dir.exists():
            for f in frames_dir.glob("frame_*.jpg"):
                f.unlink()  # delete each existing frame

        logger.info(
            f"Extracting frames at {self.target_fps} FPS from: {video_path}"
        )

        # FFmpeg command breakdown:
        # -i {video_path}        → input video
        # -vf fps={target_fps}   → video filter: output exactly N frames
        #                          per second of source video
        #                          "fps=1.0" = one frame per second
        #                          "fps=5.0" = five frames per second
        # -q:v 2                 → JPEG quality (1=best, 31=worst, 2 ≈ 95%)
        # -y                     → overwrite without asking
        # frames/frame_%06d.jpg  → output pattern:
        #                          %06d = zero-padded 6-digit frame number
        #                          frame_000001.jpg, frame_000002.jpg, ...
        output_pattern = str(frames_dir / "frame_%06d.jpg")

        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vf", f"fps={self.target_fps}",   # frame rate filter
            "-q:v", "2",                         # JPEG quality (2 = ~95%)
            "-y",
            output_pattern
        ]

        logger.info("Running FFmpeg frame extraction (this may take a moment)...")
        logger.debug(f"Command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg frame extraction failed.\n"
                f"Stderr: {result.stderr[-2000:]}"
            )

        # Count how many frames were actually extracted
        extracted_frames = sorted(frames_dir.glob("frame_*.jpg"))
        num_frames = len(extracted_frames)

        if num_frames == 0:
            raise RuntimeError(
                f"FFmpeg ran without error but no frames were saved in "
                f"{frames_dir}. Check that the input video has a video "
                f"stream and is not corrupted."
            )

        total_size_mb = sum(
            f.stat().st_size for f in extracted_frames
        ) / (1024 * 1024)

        logger.info(
            f"Frames extracted: {num_frames} frames → {frames_dir} "
            f"({total_size_mb:.1f} MB total)"
        )

        return str(frames_dir)


    # ─────────────────────────────────────────────────────────────────────
    # SECTION: Get Frame Path for a Specific Timestamp
    # ─────────────────────────────────────────────────────────────────────

    def get_frame_path_at_timestamp(self, timestamp_seconds: float) -> Optional[str]:
        """
        Given a timestamp in seconds, return the path to the closest
        extracted frame.

        This is used by the visual analysis stages (Phases 6–9).
        When PANNs detects a sound event at, say, 12.3 seconds,
        the visual stage calls this to find the frame image at that moment.

        With 1 FPS extraction, the frame for second 12.3 is frame_000012.jpg.
        With 5 FPS extraction, the frame for second 12.3 is frame_000062.jpg
        (12.3 seconds × 5 frames/second = 61.5 → frame 62).

        Returns None if the frame file doesn't exist.
        """
        # Calculate which frame number corresponds to this timestamp
        frame_number = int(timestamp_seconds * self.target_fps) + 1
        # +1 because FFmpeg starts frame numbering at 1, not 0

        frame_path = self.frames_output_dir / f"frame_{frame_number:06d}.jpg"

        if frame_path.exists():
            return str(frame_path)
        else:
            logger.warning(
                f"No frame found at timestamp {timestamp_seconds:.2f}s "
                f"(expected: {frame_path})"
            )
            return None


    def get_frame_window(
        self,
        center_timestamp: float,
        window_seconds: float = 1.0
    ) -> list[str]:
        """
        Return paths to all extracted frames within a time window
        centered on the given timestamp.

        Example: center=12.3s, window=1.0s → frames from 11.3s to 13.3s

        This is used by the visual reaction detector. When an audio event
        fires at 12.3s, we look at frames in the ±1 second window to
        check if anyone reacted.

        Returns a list of existing frame paths, sorted by time.
        """
        start_time = max(0.0, center_timestamp - window_seconds)
        end_time = center_timestamp + window_seconds

        # Calculate frame numbers for the window boundaries
        start_frame = int(start_time * self.target_fps) + 1
        end_frame = int(end_time * self.target_fps) + 1

        frame_paths = []
        for frame_num in range(start_frame, end_frame + 1):
            frame_path = self.frames_output_dir / f"frame_{frame_num:06d}.jpg"
            if frame_path.exists():
                frame_paths.append(str(frame_path))

        return sorted(frame_paths)  # sorted ensures chronological order


    # ─────────────────────────────────────────────────────────────────────
    # SECTION: Main Entry Point — Run Full Ingestion
    # ─────────────────────────────────────────────────────────────────────

    def ingest(self, video_path: str) -> ExtractionResult:
        """
        Run the full ingestion pipeline on a video file.
        This is the single method that downstream stages will call.

        Steps:
            1. Probe the video to get metadata
            2. Validate that the video has an audio stream
            3. Extract audio as 16kHz mono WAV
            4. Extract frames at configured FPS
            5. Return ExtractionResult with all paths and metadata

        Raises:
            FileNotFoundError: if video_path doesn't exist
            ValueError: if video has no audio stream
            RuntimeError: if FFmpeg extraction fails
        """
        video_path = str(video_path)

        # Step 1: Probe
        logger.info("=" * 60)
        logger.info(f"Starting ingestion for: {video_path}")
        logger.info("=" * 60)

        metadata = self.probe_video(video_path)

        # Step 2: Validate
        if not metadata["has_audio"]:
            raise ValueError(
                f"Video has no audio stream: {video_path}\n"
                "This pipeline requires audio for sound event detection."
            )

        if not metadata["has_video"]:
            raise ValueError(
                f"File has no video stream: {video_path}\n"
                "This pipeline requires video for visual reaction analysis."
            )

        # Step 3: Extract audio
        audio_path = self.extract_audio(video_path)

        # Step 4: Extract frames
        frames_dir = self.extract_frames(video_path)

        # Step 5: Count extracted frames
        num_frames = len(list(Path(frames_dir).glob("frame_*.jpg")))

        # Step 6: Build and return result
        result = ExtractionResult(
            audio_path=audio_path,
            frames_dir=frames_dir,
            duration_seconds=metadata["duration"],
            fps=metadata["fps"],
            sample_rate=self.target_sample_rate,
            num_frames_extracted=num_frames
        )

        logger.info("=" * 60)
        logger.info("Ingestion complete.")
        logger.info(f"  Audio:  {result.audio_path}")
        logger.info(f"  Frames: {result.frames_dir} ({num_frames} frames)")
        logger.info(f"  Duration: {result.duration_seconds:.1f}s")
        logger.info("=" * 60)

        return result