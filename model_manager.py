import os
import json
import glob
from typing import Dict, List, Optional, Callable

import config

# Curated GGUF catalog available for one-click download from HuggingFace Hub.
MODEL_CATALOG = [
    {
        "key": "llama-3.2-3b-instruct-q4",
        "name": "Llama 3.2 3B Instruct (Q4_K_M)",
        "repo": "unsloth/Llama-3.2-3B-Instruct-GGUF",
        "file": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "size_gb": 2.2,
        "ram_gb": 4.5,
        "tag": "Recommended",
        "desc": "Best quality/speed balance on any modern PC.",
    },
    {
        "key": "llama-3.2-1b-instruct-q4",
        "name": "Llama 3.2 1B Instruct (Q4_K_M)",
        "repo": "unsloth/Llama-3.2-1B-Instruct-GGUF",
        "file": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "size_gb": 0.85,
        "ram_gb": 2.0,
        "tag": "Fast",
        "desc": "Very fast on low-end PCs; ideal for Fast mode.",
    },
    {
        "key": "qwen2.5-3b-instruct-q4",
        "name": "Qwen2.5 3B Instruct (Q4_K_M)",
        "repo": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "file": "qwen2.5-3b-instruct-q4_k_m.gguf",
        "size_gb": 1.9,
        "ram_gb": 4.0,
        "tag": "Recommended",
        "desc": "Strong reasoning and multilingual support.",
    },
    {
        "key": "qwen2.5-1.5b-instruct-q4",
        "name": "Qwen2.5 1.5B Instruct (Q4_K_M)",
        "repo": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "file": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "size_gb": 1.0,
        "ram_gb": 2.0,
        "tag": "Fast",
        "desc": "Great speed for laptops and office PCs.",
    },
    {
        "key": "llama-3.2-8b-instruct-q4",
        "name": "Llama 3.2 8B Instruct (Q4_K_M)",
        "repo": "unsloth/Llama-3.2-8B-Instruct-GGUF",
        "file": "Llama-3.2-8B-Instruct-Q4_K_M.gguf",
        "size_gb": 4.9,
        "ram_gb": 9.0,
        "tag": "Heavy",
        "desc": "Highest quality; requires 8 GB+ RAM, GPU recommended.",
    },
]

VISION_FILENAME_HINTS = ("moondream", "mmproj", "clip", "vision", "phi-3-vision", "llava")


def _fmt_size(n: int) -> str:
    if n >= 1 << 30:
        return f"{n / (1 << 30):.1f} GB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.1f} MB"
    return f"{n} B"


class ModelManager:
    """Scans, downloads, activates and deletes local GGUF models."""

    def __init__(self):
        self.search_dirs = [os.path.abspath(d) for d in config.MODEL_DIRS]
        for d in (self.search_dirs + [os.path.abspath(config.MODEL_DOWNLOAD_DIR)]):
            os.makedirs(d, exist_ok=True)

    # ------------------------------------------------------------------ scan
    def list_models(self, include_vision: bool = False) -> List[Dict]:
        models: Dict[str, Dict] = {}
        for d in self.search_dirs:
            for path in glob.glob(os.path.join(d, "**", "*.gguf"), recursive=True):
                path = os.path.abspath(path)
                name = os.path.basename(path)
                stats = os.stat(path)
                is_vision = any(h in name.lower() for h in VISION_FILENAME_HINTS)
                if is_vision and not include_vision:
                    continue
                models[path] = {
                    "path": path,
                    "name": name,
                    "size_bytes": stats.st_size,
                    "size": _fmt_size(stats.st_size),
                    "is_vision": is_vision,
                    "in_catalog": any(m["file"] == name for m in MODEL_CATALOG),
                }
        return sorted(models.values(), key=lambda m: m["name"].lower())

    def list_main_models(self) -> List[Dict]:
        return [m for m in self.list_models() if not m["is_vision"]]

    # ------------------------------------------------------------------ active
    def active_model_path(self) -> Optional[str]:
        """Returns the active main model path (persisted selection), else the default."""
        if os.path.exists(config.ACTIVE_MODEL_FILE):
            try:
                with open(config.ACTIVE_MODEL_FILE, "r", encoding="utf-8") as f:
                    path = json.load(f).get("active")
                if path and os.path.exists(path):
                    return path
            except Exception:
                pass
        if os.path.exists(config.LLM_MODEL_PATH):
            return config.LLM_MODEL_PATH
        mains = self.list_main_models()
        return mains[0]["path"] if mains else None

    def active_model_name(self) -> str:
        path = self.active_model_path()
        return os.path.basename(path) if path else "No model"

    def set_active(self, path: str) -> bool:
        if not path or not os.path.exists(path):
            return False
        os.makedirs(os.path.dirname(config.ACTIVE_MODEL_FILE) or ".", exist_ok=True)
        tmp = config.ACTIVE_MODEL_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"active": os.path.abspath(path)}, f, indent=2)
        os.replace(tmp, config.ACTIVE_MODEL_FILE)
        return True

    # ------------------------------------------------------------------ download
    def download(self, catalog_key: str, progress: Optional[Callable[[float, str], None]] = None) -> Optional[str]:
        """Downloads a catalog model to MODEL_DOWNLOAD_DIR. Returns the local path."""
        entry = next((m for m in MODEL_CATALOG if m["key"] == catalog_key), None)
        if not entry:
            raise KeyError(f"Unknown model key: {catalog_key}")

        target = os.path.join(config.MODEL_DOWNLOAD_DIR, entry["file"])
        if os.path.exists(target):
            if progress:
                progress(1.0, "Already downloaded.")
            return os.path.abspath(target)

        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise RuntimeError("huggingface_hub is required for downloading models. Run: pip install huggingface_hub")

        os.makedirs(config.MODEL_DOWNLOAD_DIR, exist_ok=True)

        def _hook(info):
            if progress and hasattr(info, "completed") and info.b_done and info.total is not None:
                progress(info.completed / max(1, info.total), f"{entry['name']}")

        path = hf_hub_download(
            repo_id=entry["repo"],
            filename=entry["file"],
            local_dir=config.MODEL_DOWNLOAD_DIR,
            tqdm_class=None,
        )
        return os.path.abspath(path)

    def delete_model(self, path: str) -> bool:
        path = os.path.abspath(path)
        if not os.path.exists(path):
            return False
        if self.active_model_path() == path:
            raise ValueError("Cannot delete the active model. Select another model first.")
        try:
            os.remove(path)
            return True
        except OSError as e:
            raise RuntimeError(f"Failed to delete model: {e}") from e