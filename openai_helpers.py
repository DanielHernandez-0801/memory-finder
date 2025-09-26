import base64, io, json, re
from typing import List
from PIL import Image
from config import OPENAI_API_KEY, OPENAI_MODEL_TEXT, OPENAI_MODEL_VISION
from openai import OpenAI

_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

def expand_query_with_openai(q: str) -> List[str]:
    """Expand a natural query into ~5-10 concise visual keywords for CLIP."""
    if not _client: 
        return []
    prompt = f"""
    You are helping convert a user's photo search into visual keywords that a vision embedding
    model like CLIP will understand. Return 5-10 comma-separated keywords (no sentences).
    Query: "{q}"
    Only output the keywords, separated by commas.
    """
    resp = _client.chat.completions.create(
        model=OPENAI_MODEL_TEXT,
        messages=[{"role":"user","content":prompt}],
        temperature=0.2,
        max_tokens=64,
    )
    text = resp.choices[0].message.content.strip()
    # turn "beach, ocean, shore, sand" into list
    kws = [t.strip() for t in re.split(r"[,\n]", text) if t.strip()]
    return kws[:10]

def vision_tags_for_image(pil: Image.Image) -> List[str]:
    """Ask a VLM to output 8-15 tags describing scene, objects, clothing colors, setting, etc."""
    if not _client:
        return []
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    messages = [{
        "role":"user",
        "content":[
            {"type":"text","text":"List 8-15 short tags (single words or 2-word phrases) for this photo: scene type, objects, setting (indoor/outdoor), activities, clothing colors, environment (e.g., beach, mountain, city), time of day. Output JSON array of strings only."},
            {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}}
        ]
    }]
    resp = _client.chat.completions.create(
        model=OPENAI_MODEL_VISION,
        messages=messages,
        temperature=0.2,
        max_tokens=120,
        response_format={ "type":"json_object" }  # model will return {"tags":[...]} or fallback text
    )
    out = resp.choices[0].message.content
    # Be tolerant: accept either raw JSON array or {"tags":[...]} or comma text.
    try:
        data = json.loads(out)
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
        if isinstance(data, dict) and "tags" in data and isinstance(data["tags"], list):
            return [str(x).strip() for x in data["tags"] if str(x).strip()]
    except Exception:
        pass
    # crude fallback: split on commas
    return [t.strip() for t in re.split(r"[,\n]", out) if t.strip()][:15]