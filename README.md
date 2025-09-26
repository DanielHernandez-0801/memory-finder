# Family Photo RAG (Mac)

A simple local app that indexes your family photos, extracts timestamps, captions and visual embeddings, detects faces to learn who's who, and lets you search with natural language (e.g., “all pics from 2022 Cancun”, “show 2025 mountains”, “Daniel wearing a red shirt”).

**No model training required** — uses pre-trained local models (CLIP + BLIP + InsightFace). Runs on CPU or Apple Silicon GPU via PyTorch MPS if available.

---

## Features

- **Ingest a folder of photos** (JPEG/PNG/HEIC) and store:
  - EXIF timestamp (and GPS if present)
  - BLIP caption
  - CLIP image embeddings (for semantic search like “mountains”)
  - Face detection + embeddings (for person search), with name enrollment
  - Simple **red-shirt** heuristic per person crop (for queries like “Daniel wearing a red shirt”)
- **Vector search** via ChromaDB (persistent)
- **Rules-based query parser**: understands years/dates, people, colors (red shirt), places/keywords (matches path/caption), and simple boolean mixes.
- **Streamlit UI** to run entirely on your Mac.

---

## Install (macOS)

> Tested with macOS on Apple Silicon. Intel Macs should also work (slower).

1) Create and activate a fresh Python 3.10–3.12 virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

2) Install dependencies (this can take a few minutes to download models the first time):

```bash
pip install -r requirements.txt
```

> If `insightface` wheels are slow on Apple Silicon, try:
>
> ```bash
> pip install insightface==0.7.3 onnxruntime==1.17.3 onnxruntime-silicon
> ```
> The requirements include sensible defaults already; the above is just a fallback.

Create a .env file with:
> OPENAI_API_KEY=XXX
> INDEX_MODE=XXX # FAST or FULL

3) Run the app:

```bash
streamlit run ui.py
```

4) In the app:
- Click **Index ➜ Choose Photos Root** to select your photo folder (it will recurse).
- (Optional) Go to **People ➜ Enroll** to add labeled face examples for each family member (e.g., “Daniel”).
- Use the **Search** box:
  - `2022 Cancun`
  - `mountains 2025`
  - `Daniel red shirt`
  - `all pictures from July 2023`
  - `pull up all of our pictures from our 2022 Cancun trip`

Results display as thumbnails; click to open the file in Finder.

---

## Data locations

- ChromaDB persists under `./data/chroma/`
- App DB (SQLite) at `./data/app.db`
- Person embeddings at `./data/persons.json`
- Thumbnails cached at `./data/thumbs/`

---

## Notes & Limitations

- **Privacy**: Everything runs locally. No images leave your machine.
- **HEIC**: If HEIC support is missing, macOS preview-converted JPEGs work best. You can also `brew install libheif` then `pip install pillow-heif` to add HEIC; the code will use it if available.
- **Red shirt** is a simple heuristic using HSV thresholds on the torso area under a detected face. It won’t be perfect but works well enough for casual searches.
- **Speed**: First run will download models and build embeddings. Subsequent runs are faster and incremental.
- **GPU**: On Apple Silicon, PyTorch MPS is used when available for CLIP; InsightFace remains on CPU/ONNX.

---

## Uninstall / Reset

To reset the database (keep your photos intact), delete the `data/` folder:

```bash
rm -rf data
```

