import re
from datetime import datetime
from dateutil import parser as dateparser
from typing import Dict, Any
from config import INDEX_MODE  # "FULL" or "FAST"

YEAR_RE = re.compile(r"(?:19|20)\d{2}")
MONTHS = {
    m.lower(): i
    for i, m in enumerate(
        ["January","February","March","April","May","June","July","August","September","October","November","December"],
        start=1
    )
}

# Common stopwords we can safely ignore for keyword/path matching
STOPWORDS = {
    "show","me","all","pictures","photos","from","with","in","of","our","trip","wearing",
    "the","a","an","to","pull","up","at","on"
}

def parse_query(q: str) -> Dict[str, Any]:
    """
    Returns a dict: {
        'year': int|None,
        'month': int|None,
        'person': str|None,
        'red_shirt': bool,
        'keywords': [str,...],
        'date_from': None,
        'date_to': None
    }

    In FAST mode:
      - We DO NOT remove a detected person token from keywords.
      - We also force-add any detected person token back into keywords (lowercased).
      - Goal: maximize matches against file/folder paths (e.g., country/city names).
    """
    out = {
        "year": None,
        "month": None,
        "person": None,
        "red_shirt": False,
        "keywords": [],
        "date_from": None,
        "date_to": None,
    }

    if not q or not q.strip():
        return out

    q_str = q.strip()
    ql = q_str.lower()

    # Flags
    if "red shirt" in ql or "red top" in ql:
        out["red_shirt"] = True

    # Year
    years = YEAR_RE.findall(q_str)
    if years:
        # re.findall gives full matches thanks to non-capturing group
        y = re.findall(r"(?:19|20)\d{2}", q_str)
        if y:
            out["year"] = int(y[0])

    # Month
    for name, num in MONTHS.items():
        if name in ql:
            out["month"] = num
            break

    # Naive person heuristic (capitalized token following 'of'/'for', or first standalone capitalized token)
    person = None
    m = re.search(r"(?:\bof\b|\bfor\b)\s+([A-Z][a-zA-Z]+)", q_str)
    if m:
        person = m.group(1)
    else:
        caps = re.findall(r"\b([A-Z][a-zA-Z]+)\b", q_str)
        # Skip months and years
        for c in caps:
            cl = c.lower()
            if cl not in MONTHS and not re.match(r"(?:19|20)\d{2}", c):
                person = c
                break
    if person:
        out["person"] = person

    # Build keywords:
    # - break into alpha tokens from the ORIGINAL query (to keep proper words)
    # - drop months, years, and obvious stopwords
    tokens = [t for t in re.findall(r"[A-Za-z]+", q_str)]
    kw = []
    for t in tokens:
        tl = t.lower()
        if tl in STOPWORDS:
            continue
        if tl in MONTHS:
            continue
        if YEAR_RE.fullmatch(t):
            continue
        kw.append(tl)

    # In FULL mode, old behavior was to remove the person token from keywords to reduce duplication.
    # In FAST mode, we *keep* it (and even force-add) so that place-like names still match paths.
    if INDEX_MODE == "FULL":
        if person:
            kw = [w for w in kw if w != person.lower()]
    else:
        # FAST mode: ensure person (if any) is present in keywords
        if person and person.lower() not in kw:
            kw.append(person.lower())

    # Deduplicate while preserving order
    seen = set()
    keywords = []
    for w in kw:
        if w not in seen:
            keywords.append(w)
            seen.add(w)

    out["keywords"] = keywords
    return out