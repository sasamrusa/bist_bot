import json
from pathlib import Path

from bist_bot.ai_pipeline.registry import load_latest_model_bundle, save_model_bundle


def test_save_model_bundle_writes_relative_model_path(tmp_path):
    model_dir = tmp_path / "models" / "ai_registry"
    bundle = {"model": {"kind": "dummy"}, "feature_columns": ["a"], "model_backend": "test_backend"}
    model_path = save_model_bundle(bundle, model_dir)

    meta = json.loads((model_dir / "latest.json").read_text(encoding="utf-8"))
    assert meta["model_path"] == model_path.name


def test_load_latest_model_bundle_supports_legacy_bist_bot_prefix(tmp_path):
    project_root = tmp_path / "project"
    model_dir = project_root / "models" / "ai_registry"
    model_dir.mkdir(parents=True, exist_ok=True)

    bundle = {"model": {"kind": "dummy"}, "feature_columns": ["x"], "model_backend": "legacy"}
    real_path = save_model_bundle(bundle, model_dir)
    real_bundle = load_latest_model_bundle(model_dir)
    assert real_bundle.get("model_backend") == "legacy"

    # Overwrite latest.json with legacy CWD-sensitive path style.
    legacy_meta = {
        "model_path": f"bist_bot/models/ai_registry/{real_path.name}",
        "created_at": "2026-02-12T00:00:00",
        "model_backend": "legacy",
    }
    (model_dir / "latest.json").write_text(json.dumps(legacy_meta, indent=2), encoding="utf-8")

    loaded = load_latest_model_bundle(model_dir)
    assert loaded.get("model_backend") == "legacy"
