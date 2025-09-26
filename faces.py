from typing import List, Dict, Tuple
import numpy as np
import cv2
import json
from pathlib import Path
from insightface.app import FaceAnalysis
from config import FACE_PROVIDER, FACE_DET_SIZE, PERSONS_JSON
from PIL import Image

# Load face detector/recognizer
_face_app = FaceAnalysis(providers=[FACE_PROVIDER], allowed_modules=['detection','recognition'])
_face_app.prepare(ctx_id=0, det_size=(FACE_DET_SIZE, FACE_DET_SIZE))

def detect_faces(pil_image: Image.Image):
    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    faces = _face_app.get(img)  # each has bbox, normed_embedding
    results = []
    for f in faces:
        x1,y1,x2,y2 = [int(v) for v in f.bbox]
        emb = f.normed_embedding.astype("float32")
        results.append({"bbox": (x1,y1,x2,y2), "embedding": emb})
    return results

def load_persons() -> Dict[str, List[float]]:
    p = Path(PERSONS_JSON)
    if p.exists():
        return json.loads(p.read_text())
    return {}

def save_persons(d: Dict[str, List[float]]):
    Path(PERSONS_JSON).write_text(json.dumps(d, indent=2))

def register_person(name: str, face_embeddings: List[np.ndarray]):
    persons = load_persons()
    if not face_embeddings:
        return
    avg = np.mean(np.stack(face_embeddings, axis=0), axis=0)
    avg = avg / (np.linalg.norm(avg) + 1e-9)
    persons[name] = avg.tolist()
    save_persons(persons)

def recognize(embedding: np.ndarray, thr: float = 0.4) -> Tuple[str, float]:
    # cosine distance threshold - smaller is closer; we use similarity
    persons = load_persons()
    if not persons:
        return "", 0.0
    best_name, best_sim = "", -1.0
    for name, ref in persons.items():
        refv = np.array(ref, dtype="float32")
        sim = float(np.dot(embedding, refv) / (np.linalg.norm(embedding)*np.linalg.norm(refv) + 1e-9))
        if sim > best_sim:
            best_sim, best_name = sim, name
    if best_sim >= (1.0 - thr):  # similarity close to 1
        return best_name, best_sim
    return "", best_sim

def red_shirt_ratio(pil_image: Image.Image, bbox):
    # crude heuristic: crop a rectangle below the face (torso zone)
    x1,y1,x2,y2 = bbox
    w = x2 - x1
    h = y2 - y1
    H,W = pil_image.size[1], pil_image.size[0]
    # define torso crop: start at y2, extend 1.2 * face height downwards
    yA = max(0, y2)
    yB = min(H, y2 + int(1.2*h))
    xA = max(0, x1 - int(0.25*w))
    xB = min(W, x2 + int(0.25*w))
    if yB <= yA or xB <= xA:
        return 0.0
    crop = pil_image.crop((xA,yA,xB,yB))
    bgr = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # red ranges in HSV (two ranges around 0 and 180)
    lower1 = np.array([0, 70, 50]); upper1 = np.array([10,255,255])
    lower2 = np.array([170,70,50]); upper2 = np.array([180,255,255])
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)
    ratio = float(np.count_nonzero(mask)) / float(mask.size)
    return ratio
