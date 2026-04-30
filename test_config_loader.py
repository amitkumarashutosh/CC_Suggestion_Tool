import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

from src.config_loader import ConfigLoader

def test_config_loader():
    print("=" * 50)
    print("PHASE 13 VERIFICATION TEST")
    print("=" * 50)

    # Load config
    print("\n[1/4] Loading config.yaml...")
    loader = ConfigLoader("config.yaml")
    config = loader.load()
    print("  → Config loaded and validated")

    # Check all sections present
    print("\n[2/4] Verifying all sections...")
    sections = [
        "paths", "ingestion", "audio_processor", "sound_detector",
        "event_filter", "frame_extractor", "face_analyzer",
        "pose_analyzer", "visual_scorer", "decision_engine",
        "label_generator", "srt_writer"
    ]
    for section in sections:
        assert hasattr(config, section), f"FAIL: Missing section: {section}"
        print(f"  PASS: {section}")

    # Check key values match YAML
    print("\n[3/4] Verifying key values...")
    assert config.decision_engine.cc_threshold == 0.60, \
        f"FAIL: cc_threshold={config.decision_engine.cc_threshold}"
    print(f"  PASS: cc_threshold = {config.decision_engine.cc_threshold}")

    assert config.ingestion.extraction_fps == 1.0, \
        f"FAIL: extraction_fps={config.ingestion.extraction_fps}"
    print(f"  PASS: extraction_fps = {config.ingestion.extraction_fps}")

    assert config.sound_detector.device == "cpu", \
        f"FAIL: device={config.sound_detector.device}"
    print(f"  PASS: device = {config.sound_detector.device}")

    assert config.event_filter.suppress_sustained_music == True, \
        "FAIL: suppress_sustained_music should be True"
    print(f"  PASS: suppress_sustained_music = True")

    # Test profile switching
    print("\n[4/4] Testing profile switching...")
    import yaml
    with open("config.yaml") as f:
        raw = yaml.safe_load(f)

    # Temporarily test aggressive profile
    raw["active_profile"] = "aggressive"
    import tempfile, os
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.yaml', delete=False
    ) as tmp:
        yaml.dump(raw, tmp)
        tmp_path = tmp.name

    try:
        tmp_loader = ConfigLoader(tmp_path)
        aggressive_config = tmp_loader.load()
        assert aggressive_config.decision_engine.cc_threshold == 0.52, \
            f"FAIL: aggressive cc_threshold should be 0.52"
        print(f"  PASS: aggressive profile: "
              f"cc_threshold={aggressive_config.decision_engine.cc_threshold}")
    finally:
        os.unlink(tmp_path)

    # Print summary
    print()
    loader.print_summary(config)

    print(f"  All pipeline parameters now controlled by config.yaml")
    print(f"  No code changes needed to tune the system")

if __name__ == "__main__":
    test_config_loader()