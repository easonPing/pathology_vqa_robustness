from __future__ import annotations

import os
from pathlib import Path


DEFAULT_HF_HOME = Path("/bigtemp/efk4ps/huggingface")


def configure_huggingface_cache(hf_home: str | Path) -> Path:
    """Set all Hugging Face caches before Transformers is imported."""
    root = Path(hf_home).expanduser().resolve()
    home = Path.home().resolve()
    if root == home or home in root.parents:
        raise ValueError(
            f"Hugging Face cache must not be under the home quota: {root}"
        )

    hub = root / "hub"
    xet = root / "xet"
    hub.mkdir(parents=True, exist_ok=True)
    xet.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(root)
    os.environ["HF_HUB_CACHE"] = str(hub)
    os.environ["HF_XET_CACHE"] = str(xet)
    return root

