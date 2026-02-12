from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import joblib


def save_model_bundle(bundle: Dict[str, Any], model_dir: Path) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = model_dir / f"ai_model_{ts}.joblib"
    joblib.dump(bundle, model_path)

    latest_meta = {
        # Keep path relative to model_dir so loading does not depend on CWD.
        "model_path": model_path.name,
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
    raw_model_path = str(meta.get("model_path", "")).strip()
    if not raw_model_path:
        raise FileNotFoundError(f"Invalid latest metadata at {latest_path}: missing model_path")
    model_path = _resolve_model_path(raw_model_path, model_dir)
    if model_path is None:
        raise FileNotFoundError(
            f"Latest model file missing. latest.json={latest_path} model_path={raw_model_path}"
        )
    return load_model_bundle(model_path)


def _resolve_model_path(raw_model_path: str, model_dir: Path) -> Path | None:
    configured = Path(raw_model_path)
    model_dir_resolved = model_dir.resolve()
    project_root = model_dir_resolved.parent.parent if len(model_dir_resolved.parents) >= 2 else model_dir_resolved.parent

    candidates: List[Path] = []
    if configured.is_absolute():
        candidates.append(configured)
    else:
        # 1) relative to current working directory
        candidates.append(configured)
        # 2) relative to model directory
        candidates.append(model_dir_resolved / configured)
        # 3) relative to project root (handles "models/..." style paths)
        candidates.append(project_root / configured)

        # 4) handle legacy prefix: "bist_bot/models/..."
        parts = configured.parts
        if parts and parts[0].lower() == "bist_bot" and len(parts) > 1:
            stripped = Path(*parts[1:])
            candidates.append(project_root / stripped)
            candidates.append(model_dir_resolved / stripped)

    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
    return None
