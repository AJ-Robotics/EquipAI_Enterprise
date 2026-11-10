import os
import math
import sys as _sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def _base_dir():
    """Base directory for persistent data. For a frozen EXE this is the folder
    that contains the executable, so models/stores live next to it (portable)."""
    if getattr(_sys, "frozen", False):
        return os.path.dirname(os.path.abspath(_sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


BASE_DIR = _base_dir()


def _resolve(rel):
    """Resolves a (possibly relative) path against the app base directory."""
    if os.path.isabs(rel):
        return rel
    return os.path.join(BASE_DIR, rel)


def _int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _bool(name, default):
    v = os.environ.get(name)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "on")


def ram_free_gb():
    """Returns approximate free physical RAM in GiB using psutil or Windows API."""
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 ** 3)
    except ImportError:
        pass

    if os.name == "nt":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            m = MEMORYSTATUSEX()
            m.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                return m.ullAvailPhys / (1024 ** 3)
        except Exception:
            pass

    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except Exception:
        pass

    return 0.0


CPU_COUNT = os.cpu_count() or 4

# --- Model paths (resolved against BASE_DIR so the EXE is portable) ---
LLM_MODEL_PATH = os.environ.get("EQPAI_LLM_MODEL") or _resolve("llm_model/llama-3.2-3b-instruct-q4_k_m.gguf")
VISION_MODEL_PATH = os.environ.get("EQPAI_VISION_MODEL") or _resolve("llm_model/moondream2-text-model-f16.gguf")
CLIP_MODEL_PATH = os.environ.get("EQPAI_CLIP_MODEL") or _resolve("llm_model/moondream2-mmproj-f16.gguf")
EMBEDDING_MODEL_PATH = os.environ.get("EQPAI_EMBEDDING_MODEL") or _resolve("embedding_model")

# --- CPU thread budgets ---
# Higher is faster per query but starves the rest of the machine and increases
# memory pressure on iGPU-less PCs. Leave one core free for the UI by default.
LLM_THREADS = _int("EQPAI_LLM_THREADS", max(2, min(6, CPU_COUNT - 1)))
VISION_THREADS = _int("EQPAI_VISION_THREADS", max(2, min(4, CPU_COUNT - 1)))
EMBED_THREADS = _int("EQPAI_EMBED_THREADS", min(4, CPU_COUNT))

# --- Model context / batch ---
LLM_CTX = _int("EQPAI_LLM_CTX", 4096)
VISION_CTX = _int("EQPAI_VISION_CTX", 2048)
BATCH_SIZE = _int("EQPAI_BATCH_SIZE", 256)          # lower than default speeds up CPU llama.cpp
VISION_BATCH = _int("EQPAI_VISION_BATCH", 256)
EMBED_BATCH = _int("EQPAI_EMBED_BATCH", 32)

# --- Generation ---
TEMPERATURE = _float("EQPAI_TEMPERATURE", 0.2)      # 0.0 can loop on some llama.cpp builds
REPEAT_PENALTY = _float("EQPAI_REPEAT_PENALTY", 1.1)
TOP_K = _int("EQPAI_TOP_K", 40)
TOP_P = _float("EQPAI_TOP_P", 0.9)
MAX_TOKENS = _int("EQPAI_MAX_TOKENS", 900)
MAX_PROMPT_TOKENS = _int("EQPAI_MAX_PROMPT_TOKENS", 2900)
MAX_HISTORY_TURNS = _int("EQPAI_MAX_HISTORY_TURNS", 5)

# --- Features ---
USE_GPU = _bool("EQPAI_USE_GPU", False)
USE_VISION = _bool("EQPAI_USE_VISION", True)
VISION_MIN_FREE_GB = _float("EQPAI_VISION_MIN_FREE_GB", 6.0)  # f16 vision stack needs ~5.5 GB
USE_HYBRID = _bool("EQPAI_HYBRID", True)

# --- Retrieval ---
RETRIEVAL_K = _int("EQPAI_RETRIEVAL_K", 5)
RETRIEVAL_THRESHOLD = _float("EQPAI_RETRIEVAL_THRESHOLD", 0.15)

# --- Ingestion ---
CHUNK_SIZE = _int("EQPAI_CHUNK_SIZE", 500)
CHUNK_OVERLAP = _int("EQPAI_CHUNK_OVERLAP", 50)

# --- Storage ---
VECTOR_STORE_PATH = os.environ.get("EQPAI_VECTOR_STORE") or _resolve("vector_store.pkl")
FEEDBACK_FILE = os.environ.get("EQPAI_FEEDBACK_FILE") or _resolve("feedback_data.json")
FEEDBACK_VECTOR_STORE = os.environ.get("EQPAI_FEEDBACK_VECTORS") or _resolve("feedback_vectors.pkl")
SESSIONS_DIR = os.environ.get("EQPAI_SESSIONS_DIR") or _resolve("chat_sessions")
MEDIA_DIR = os.environ.get("EQPAI_MEDIA_DIR") or _resolve("media")
TEMP_UPLOAD_DIR = os.environ.get("EQPAI_TEMP_UPLOAD_DIR") or _resolve("temp_uploads")
ANALYTICS_FILE = os.environ.get("EQPAI_ANALYTICS_FILE") or _resolve("analytics.json")

# --- Model management ---
MODEL_DIRS = [
    d if os.path.isabs(d) else _resolve(d)
    for d in os.environ.get("EQPAI_MODEL_DIRS", "llm_model;models/llm;models").split(";") if d
]
MODEL_DOWNLOAD_DIR = os.environ.get("EQPAI_MODEL_DOWNLOAD_DIR") or _resolve("models/llm")
ACTIVE_MODEL_FILE = os.environ.get("EQPAI_ACTIVE_MODEL_FILE") or _resolve("models/active_model.json")

# --- Response modes (fast / balanced / research) ---
DEFAULT_RESPONSE_MODE = os.environ.get("EQPAI_DEFAULT_MODE", "balanced")

RESPONSE_MODES = {
    # "fast": minimal context + short answer for quick triage
    "fast": {
        "label": "Fast",
        "icon": "⚡",
        "k": 2,
        "max_tokens": 300,
        "history_turns": 0,
        "use_feedback": False,
        "usage": "Real-time triage on the shop floor. Prioritizes speed and brevity.",
    },
    # "balanced": default professional troubleshooting
    "balanced": {
        "label": "Balanced",
        "icon": "⚖️",
        "k": 5,
        "max_tokens": 700,
        "history_turns": 4,
        "use_feedback": True,
        "usage": "Standard step-by-step diagnostics grounded in the knowledge base.",
    },
    # "research": deep retrieval + longer, cited reports
    "research": {
        "label": "Research",
        "icon": "🔬",
        "k": 8,
        "max_tokens": 1200,
        "history_turns": 6,
        "use_feedback": True,
        "usage": "Deep analysis across many documents, better for RCA / reports.",
    },
}


def response_mode(mode: str) -> dict:
    return RESPONSE_MODES.get(mode, RESPONSE_MODES["balanced"])

# --- Thresholds ---
MAX_EVENTS = _int("EQPAI_MAX_EVENTS", 5000)   # analytics ring-buffer cap


def resolve_threads(preferred):
    """Constrains a requested thread count to actual capability."""
    return max(1, min(int(preferred), CPU_COUNT))


def tokens_for_characters(characters):
    """Rough 1 token ~ 4 chars estimate; intentionally pessimistic for CPU safety."""
    return math.ceil(characters / 3.5)