# batch_process.py
# =============================================================================
# Batch Video Processor
# =============================================================================
# Process multiple video files with one command.
#
# Usage:
#   python3 batch_process.py --input_dir videos/ --output_dir srt_output/
#   python3 batch_process.py --input_dir videos/ --profile aggressive
#   python3 batch_process.py --file_list videos.txt
# =============================================================================

import argparse
import subprocess
import sys
import time
from pathlib import Path


SUPPORTED_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}


def find_videos(input_dir: str) -> list:
    """Find all supported video files in a directory."""
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        return []

    videos = []
    for ext in SUPPORTED_EXTENSIONS:
        videos.extend(input_path.glob(f"*{ext}"))
        videos.extend(input_path.glob(f"*{ext.upper()}"))

    return sorted(set(videos))


def process_batch(
    video_paths: list,
    output_dir: str,
    extra_args: list = None
) -> dict:
    """
    Process a list of video files through the pipeline.

    For each video:
      - Runs main.py with --video and --output flags
      - Captures success/failure
      - Reports timing

    Returns a summary dict with results per video.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "total": len(video_paths),
        "succeeded": [],
        "failed": [],
        "skipped": []
    }

    print(f"\n{'═' * 60}")
    print(f"BATCH PROCESSING: {len(video_paths)} videos")
    print(f"Output directory: {output_dir}")
    print(f"{'═' * 60}")

    batch_start = time.time()

    for i, video_path in enumerate(video_paths, 1):
        video_name = video_path.stem
        srt_output = output_path / f"{video_name}.srt"

        print(f"\n[{i}/{len(video_paths)}] Processing: {video_path.name}")

        # Skip if SRT already exists (re-run safety)
        if srt_output.exists():
            print(f"  SKIP: SRT already exists: {srt_output}")
            results["skipped"].append(str(video_path))
            continue

        # Build command
        cmd = [
            sys.executable, "main.py",
            "--video", str(video_path),
            "--output", str(srt_output),
        ]
        if extra_args:
            cmd.extend(extra_args)

        # Run pipeline for this video
        video_start = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per video
            )
            elapsed = time.time() - video_start

            if result.returncode == 0:
                # Extract CC count from output
                cc_count = 0
                for line in result.stdout.split("\n"):
                    if "CC Annotations Generated:" in line:
                        try:
                            cc_count = int(line.split(":")[1].strip())
                        except Exception:
                            pass

                print(
                    f"  ✓ Done ({elapsed:.1f}s) — "
                    f"{cc_count} CC annotations → {srt_output.name}"
                )
                results["succeeded"].append({
                    "video": str(video_path),
                    "srt": str(srt_output),
                    "time": elapsed,
                    "cc_count": cc_count
                })
            else:
                print(f"  ✗ FAILED ({elapsed:.1f}s)")
                # Print last 10 lines of stderr for diagnosis
                stderr_lines = result.stderr.strip().split("\n")
                for line in stderr_lines[-10:]:
                    print(f"    {line}")
                results["failed"].append({
                    "video": str(video_path),
                    "error": result.stderr[-500:]
                })

        except subprocess.TimeoutExpired:
            print(f"  ✗ TIMEOUT (>300s) — video may be too long")
            results["failed"].append({
                "video": str(video_path),
                "error": "timeout"
            })
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            results["failed"].append({
                "video": str(video_path),
                "error": str(e)
            })

    # Final summary
    total_time = time.time() - batch_start
    total_cc = sum(
        r["cc_count"] for r in results["succeeded"]
    )

    print(f"\n{'═' * 60}")
    print(f"BATCH COMPLETE")
    print(f"{'═' * 60}")
    print(f"  Total videos:    {results['total']}")
    print(f"  Succeeded:       {len(results['succeeded'])}")
    print(f"  Failed:          {len(results['failed'])}")
    print(f"  Skipped:         {len(results['skipped'])}")
    print(f"  Total CC labels: {total_cc}")
    print(f"  Total time:      {total_time:.1f}s")
    print(
        f"  Avg per video:   "
        f"{total_time/max(len(results['succeeded']), 1):.1f}s"
    )

    if results["failed"]:
        print(f"\n  Failed videos:")
        for f in results["failed"]:
            print(f"    ✗ {Path(f['video']).name}: {f['error'][:80]}")

    print(f"{'═' * 60}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Batch process multiple videos through the CC pipeline"
    )
    parser.add_argument(
        "--input_dir", "-i",
        type=str,
        help="Directory containing video files to process"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="data/output/batch",
        help="Directory where SRT files will be saved"
    )
    parser.add_argument(
        "--file_list", "-f",
        type=str,
        help="Text file with one video path per line"
    )
    parser.add_argument(
        "--profile", "-p",
        type=str,
        choices=["aggressive", "conservative", "music_video"],
        help="Pipeline profile to use for all videos"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        help="CC threshold override for all videos"
    )
    parser.add_argument(
        "--skip-visual",
        action="store_true",
        help="Skip visual analysis (faster, audio-only mode)"
    )

    args = parser.parse_args()

    # Collect video paths
    video_paths = []

    if args.input_dir:
        video_paths = find_videos(args.input_dir)
        if not video_paths:
            print(f"No supported video files found in: {args.input_dir}")
            print(f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}")
            sys.exit(1)

    elif args.file_list:
        file_list_path = Path(args.file_list)
        if not file_list_path.exists():
            print(f"File list not found: {args.file_list}")
            sys.exit(1)
        with open(file_list_path) as f:
            video_paths = [
                Path(line.strip())
                for line in f
                if line.strip() and not line.startswith("#")
            ]

    else:
        print("ERROR: Provide --input_dir or --file_list")
        parser.print_help()
        sys.exit(1)

    # Build extra args to pass to main.py
    extra_args = []
    if args.profile:
        extra_args.extend(["--profile", args.profile])
    if args.threshold:
        extra_args.extend(["--threshold", str(args.threshold)])
    if args.skip_visual:
        extra_args.append("--skip-visual")

    # Run batch
    process_batch(video_paths, args.output_dir, extra_args)


if __name__ == "__main__":
    main()