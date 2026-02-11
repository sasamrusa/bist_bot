from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import joblib


def save_model_bundle(bundle: Dict[str, Any], model_dir: Path) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = model_dir / f"ai_model_{ts}.joblib"
    joblib.dump(bundle, model_path)

    latest_meta = {
        "model_path": str(model_path),
        "created_at": datetime.now().isoformat(),
        "model_backend": bundle.get("model_backend", "unknown"),
    }
    (model_dir / "latest.json").write_text(json.dumps(latest_meta, indent=2), encoding="utf-8")
    return model_path


def load_model_bundle(path: Path) -> Dict[str, Any]:
    return joblib.load(path)


def load_latest_model_bundle(model_dir: Path) -> Dict[str, Any]:
    latest_path = model_dir / "latest.json"
    if not latest_path.exists():
        raise FileNotFoundError(f"No latest model metadata found at {latest_path}")
    meta = json.loads(latest_path.read_text(encoding="utf-8"))
    model_path = Path(meta["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(f"Latest model file missing: {model_path}")
    return load_model_bundle(model_path)

