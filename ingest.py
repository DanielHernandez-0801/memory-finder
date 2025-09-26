from pathlib import Path
from typing import Optional, List
from PIL import Image
from datetime import datetime
from dateutil import parser as dateparser
import os, json, hashlib, re

from config import SUPPORTED_EXTS, THUMBS_DIR, INDEX_MODE
from db import get_session, Image as ImageRow, Face as FaceRow
from config import CHROMA_DIR
import chromadb

# Optional imports used only in FULL mode
if INDEX_MODE == "FULL":
    from captions import caption_image
    from embeddings import image_embedding
    from faces import detect_faces, recognize, red_shirt_ratio
    from config import OPENAI_API_KEY, USE_OPENAI_VISION_TAGS
    if OPENAI_API_KEY and USE_OPENAI_VISION_TAGS:
        from openai_helpers import vision_tags_for_image

# Initialize Chroma (only used when FULL)
client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = client.get_or_create_collection(name="photos")

def _read_exif_ts(path: Path) -> Optional[datetime]:
    try:
        img = Image.open(path)
        exif = img.getexif()
        if 36867 in exif:  # DateTimeOriginal
            return dateparser.parse(exif.get(36867))
    except Exception:
        pass
    try:
        return datetime.fromtimestamp(path.stat().st_mtime)
    except Exception:
        return None

def _read_gps(path: Path) -> Optional[str]:
    return None  # placeholder; add if you want GPS

def _thumb_path(path: Path) -> Path:
    h = hashlib.md5(str(path).encode()).hexdigest()[:12]
    return THUMBS_DIR / f"{h}.jpg"

def _ensure_thumb(path: Path, max_side=400) -> Path:
    out = _thumb_path(path)
    if out.exists():
        return out
    im = Image.open(path).convert("RGB")
    w, h = im.size
    scale = max_side / max(w, h)
    if scale < 1.0:
        im = im.resize((int(w * scale), int(h * scale)))
    im.save(out, "JPEG", quality=85)
    return out

_TOKEN_SPLIT = re.compile(r"[^A-Za-z0-9]+")

def _path_tokens(p: Path) -> List[str]:
    parts = list(p.parts)
    toks: List[str] = []
    for part in parts:
        for t in _TOKEN_SPLIT.split(part):
            t = t.strip()
            if len(t) >= 2:
                toks.append(t)
    return toks

def ingest_folder(root: str):
    """
    FULL mode:
        - EXIF ts, caption (BLIP), CLIP image embed -> Chroma, faces, red shirt, optional OpenAI tags
    FAST mode:
        - EXIF ts, basic dims (cheap), store path tokens into 'tags' JSON for SQL LIKE search
        - No CLIP, no BLIP, no faces, no OpenAI calls
    """
    rootp = Path(root).expanduser()
    sess = get_session()

    paths = []
    for p in rootp.rglob("*"):
        if p.suffix.lower() in SUPPORTED_EXTS:
            paths.append(p)

    added = 0
    for p in paths:
        if sess.query(ImageRow).filter_by(path=str(p)).first():
            continue

        # Open only to get dims & thumbs quickly (both modes need thumb)
        try:
            pil = Image.open(p).convert("RGB")
        except Exception:
            continue

        ts = _read_exif_ts(p)
        gps = _read_gps(p)
        w, h = pil.size

        caption = None
        tags_json = None
        clip_emb = None

        if INDEX_MODE == "FULL":
            # Caption (can be slow)
            try:
                caption = caption_image(pil)
            except Exception:
                caption = None

            # Image embedding (required for vector search)
            try:
                emb = image_embedding(pil)
                clip_emb = emb.tolist()
            except Exception:
                clip_emb = None

            # Add to Chroma (even if caption is None; doc text can be empty)
            if clip_emb is not None:
                doc_id = str(p)
                collection.add(documents=[caption or ""], embeddings=[clip_emb], ids=[doc_id])
            else:
                doc_id = None

            # Faces
            faces = []
            try:
                faces = detect_faces(pil)
            except Exception:
                faces = []

            # Optional OpenAI vision tags
            if os.getenv("OPENAI_API_KEY") and os.getenv("USE_OPENAI_VISION_TAGS", "").lower() != "false":
                try:
                    tag_list = vision_tags_for_image(pil)
                    if tag_list:
                        tags_json = json.dumps(tag_list)
                except Exception:
                    tags_json = None

            row = ImageRow(
                path=str(p), ts=ts, gps=gps, width=w, height=h,
                caption=caption, clip_id=str(p) if clip_emb is not None else None, tags=tags_json
            )
            sess.add(row)
            sess.commit()

            # Persist faces
            for f in faces:
                name, sim = recognize(f["embedding"])
                rr = red_shirt_ratio(pil, f["bbox"])
                fr = FaceRow(image_id=row.id,
                             person_name=name or None,
                             bbox=",".join(map(str, f["bbox"])),
                             red_ratio=rr)
                sess.add(fr)
            sess.commit()

        else:
            # FAST MODE:
            # Use cheap tokens from path/folders/filename as "tags" for LIKE search later
            toks = _path_tokens(p)
            if toks:
                tags_json = json.dumps(sorted(set(toks), key=str.lower))

            row = ImageRow(
                path=str(p), ts=ts, gps=gps, width=w, height=h,
                caption=None, clip_id=None, tags=tags_json
            )
            sess.add(row)
            sess.commit()

        _ensure_thumb(p)
        added += 1

    return added

def enroll_person_from_photos(name: str, files: List[str]):
    # Person enrollment only makes sense in FULL mode where we use faces.
    if INDEX_MODE != "FULL":
        return
    from faces import register_person, detect_faces
    embs = []
    for fp in files:
        try:
            pil = Image.open(fp).convert("RGB")
            det = detect_faces(pil)
            if not det:
                continue
            det = sorted(det, key=lambda d: (d["bbox"][2]-d["bbox"][0])*(d["bbox"][3]-d["bbox"][1]), reverse=True)
            embs.append(det[0]["embedding"])
        except Exception:
            continue
    register_person(name, embs)