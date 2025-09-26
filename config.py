import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

APP_DIR = Path(__file__).parent.resolve()
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

CHROMA_DIR = DATA_DIR / "chroma"
THUMBS_DIR = DATA_DIR / "thumbs"
THUMBS_DIR.mkdir(exist_ok=True)

SQLITE_PATH = DATA_DIR / "app.db"
PERSONS_JSON = DATA_DIR / "persons.json"

# ===== Indexing mode =====
# FULL = vision-heavy (CLIP, BLIP, faces, optional OpenAI tags)
# FAST = metadata-only (timestamp + path tokens). No embeds, faces, captions, tags.
INDEX_MODE = (os.getenv("INDEX_MODE") or "FULL").strip().upper()
if INDEX_MODE not in {"FULL", "FAST"}:
    INDEX_MODE = "FULL"

# CLIP model (open_clip)
CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"

# BLIP captioning
BLIP_MODEL = "Salesforce/blip-image-captioning-base"

# Face recognition
FACE_PROVIDER = "onnxruntime"  # or "cpu"
FACE_DET_SIZE = 640

SUPPORTED_EXTS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"
}

# OpenAI (optional)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_TEXT = "gpt-4o-mini"   # for query expansion
OPENAI_MODEL_VISION = "gpt-4o-mini" # for tagging

# Flags (auto-downgrade in FAST mode)
USE_OPENAI_EXPAND_QUERY = True if INDEX_MODE == "FULL" else False
USE_OPENAI_VISION_TAGS = True if INDEX_MODE == "FULL" else False