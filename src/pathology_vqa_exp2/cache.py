from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HF_HOME = PROJECT_ROOT / "models" / "huggingface"


def configure_huggingface_cache(hf_home: str | Path) -> Path:
    """Set all Hugging Face caches before Transformers is imported."""
    root = Path(hf_home).expanduser().resolve()
    home = Path.home().resolve()
    project_models = (PROJECT_ROOT / "models").resolve()
    is_repo_local_model_cache = (
        root == project_models or project_models in root.parents
    )
    if not is_repo_local_model_cache and (root == home or home in root.parents):
        raise ValueError(
            "Hugging Face cache must be under this repository's models/ "
            f"directory when using the home filesystem: {root}"
        )

    hub = root / "hub"
    xet = root / "xet"
    hub.mkdir(parents=True, exist_ok=True)
    xet.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(root)
    os.environ["HF_HUB_CACHE"] = str(hub)
    os.environ["HF_XET_CACHE"] = str(xet)
    return root
