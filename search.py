from typing import List, Dict, Any
from db import get_session, Image as ImageRow, Face as FaceRow
from sqlalchemy import extract, or_
import chromadb
import json
from config import CHROMA_DIR, INDEX_MODE

# Conditional import for CLIP text embeddings
if INDEX_MODE == "FULL":
    from embeddings import text_embedding
else:
    text_embedding = None  # type: ignore

client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = client.get_or_create_collection(name="photos")

def _sql_like_filters(kws: List[str]):
    """Generate SQLAlchemy LIKE conditions for path/caption/tags JSON (stored as text)."""
    likes = []
    for kw in kws:
        pat = f"%{kw}%"
        likes.append(ImageRow.path.ilike(pat))
        likes.append(ImageRow.caption.ilike(pat))  # safe even if NULL
        likes.append(ImageRow.tags.ilike(pat))     # tags is JSON string
    return or_(*likes) if likes else None

def search(qobj: Dict[str, Any], k: int = 100) -> List[ImageRow]:
    sess = get_session()

    base = sess.query(ImageRow)
    if qobj.get("year"):
        base = base.filter(extract('year', ImageRow.ts) == qobj["year"])
    if qobj.get("month"):
        base = base.filter(extract('month', ImageRow.ts) == qobj["month"])

    person = qobj.get("person")
    red = qobj.get("red_shirt")

    # Person / red-shirt only available in FULL mode (faces exist)
    if INDEX_MODE == "FULL" and (person or red):
        from db import Face as FaceRow
        sub = sess.query(FaceRow.image_id)
        if person:
            sub = sub.filter(FaceRow.person_name == person)
        if red:
            sub = sub.filter(FaceRow.red_ratio >= 0.06)
        sub = sub.distinct().subquery()
        base = base.filter(ImageRow.id.in_(sub))

    imgs = base.all()

    kws = qobj.get("keywords", [])
    if not kws:
        return imgs[:k]

    # FULL: prefer CLIP vector search if we have text_embedding
    if INDEX_MODE == "FULL" and text_embedding is not None:
        try:
            text = " ".join(kws)
            qemb = text_embedding(text).tolist()
            res = collection.query(query_embeddings=[qemb], n_results=k)
            idset = set(res["ids"][0])
            ranked = [im for im in imgs if im.clip_id in idset] if imgs else [
                sess.query(ImageRow).filter(ImageRow.clip_id == i).first() for i in res["ids"][0]
            ]
            ranked = [r for r in ranked if r]
            # Boost by tag overlap if present
            kwset = {w.lower() for w in kws}
            def score(im):
                s = 1.0
                if im.tags:
                    try:
                        tlist = json.loads(im.tags)
                        if any((t.lower() in kwset) for t in tlist):
                            s += 0.5
                    except Exception:
                        pass
                return s
            ranked.sort(key=score, reverse=True)
            return ranked[:k]
        except Exception:
            # fall back to LIKE below
            pass

    # FAST (or FULL fallback): SQL LIKE across path/caption/tags
    cond = _sql_like_filters(kws)
    if cond is not None:
        imgs2 = base.filter(cond).all()
        return imgs2[:k]
    return imgs[:k]