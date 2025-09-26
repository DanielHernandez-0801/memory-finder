import streamlit as st
from pathlib import Path
from PIL import Image
from config import THUMBS_DIR, SUPPORTED_EXTS
from ingest import ingest_folder, enroll_person_from_photos
from query import parse_query
from search import search
from db import get_session, Image as ImageRow
import os, subprocess, platform
import json

st.set_page_config(page_title="Family Photo RAG", layout="wide")

# ---- Session defaults ----
if "last_query" not in st.session_state:
    st.session_state["last_query"] = ""
if "last_results" not in st.session_state:
    st.session_state["last_results"] = []
# indices requested to open/reveal after rerun
if "open_request" not in st.session_state:
    st.session_state["open_request"] = None
if "reveal_request" not in st.session_state:
    st.session_state["reveal_request"] = None

st.title("📸 Family Photo RAG")

tab1, tab2, tab3 = st.tabs(["Index", "People", "Search"])

# ---------------- Index Tab ----------------
with tab1:
    st.subheader("Index your photo library")
    photo_root = st.text_input("Photos root folder", value=str(Path.home() / "Pictures"))
    if st.button("Index"):
        with st.spinner("Indexing... (first run downloads models; be patient)"):
            added = ingest_folder(photo_root)
        st.success(f"Added {added} new photos")

# ---------------- People Tab ----------------
with tab2:
    st.subheader("Enroll / Update a Person")
    name = st.text_input("Person name (e.g., Daniel)")
    files = st.file_uploader(
        "Upload 1–10 images of this person (close-up face preferred)",
        type=["jpg", "jpeg", "png", "webp", "heic", "heif"],
        accept_multiple_files=True,
    )
    if st.button("Enroll person"):
        if not name or not files:
            st.warning("Provide name and at least one image")
        else:
            tmpdir = Path(".tmp_enroll")
            tmpdir.mkdir(exist_ok=True)
            paths = []
            for f in files:
                p = tmpdir / f.name
                p.write_bytes(f.read())
                paths.append(str(p))
            count = len(paths)
            enroll_person_from_photos(name, paths)
            for p in paths:
                try:
                    os.remove(p)
                except Exception:
                    pass
            st.success(f"Enrolled {name} from {count} image(s).")

# ---------------- Search Tab ----------------
with tab3:
    st.subheader("Search")

    def render_results(items):
        if not items:
            st.info("No results.")
            return

        cols = st.columns(5)
        for i, it in enumerate(items):
            # Build / ensure thumb (uses the hash-based path from ingest)
            from ingest import _thumb_path, _ensure_thumb  # local import to avoid circulars at top-level

            path = Path(it["path"])
            thumb = _thumb_path(path)
            _ensure_thumb(path)

            try:
                img = Image.open(thumb)
            except Exception:
                # Fallback to original if thumb missing/corrupt
                img = Image.open(path).convert("RGB")

            with cols[i % 5]:
                st.image(img, caption=f"{it['name']}\n{it['ts']}", use_column_width=True)
                c1, c2 = st.columns([1, 1])
                with c1:
                    if st.button("Open", key=f"open_{i}_{it['name']}"):
                        # Defer actual open until after rerun completes
                        st.session_state["open_request"] = i
                with c2:
                    if st.button("Reveal", key=f"reveal_{i}_{it['name']}"):
                        # Defer reveal-in-Finder until after rerun
                        st.session_state["reveal_request"] = i

    q = st.text_input(
        "Type a query (e.g., '2022 Cancun', 'mountains 2025', 'all pictures from July 2023')",
        value=st.session_state.get("last_query", ""),
    )

    run_search = st.button("Run search")
    if run_search and q.strip():
        qobj = parse_query(q)
        qobj["raw_query"] = q
        results = search(qobj, k=200)

        # Persist for reruns so UI doesn't clear on button clicks
        # st.session_state["last_query"] = q
        st.session_state["last_results"] = [
            {"path": im.path, "ts": str(im.ts) if im.ts else "", "name": Path(im.path).name}
            for im in results
        ]

        st.caption(f"Parsed: {qobj}")

    # Always render from session (sticky results across reruns)
    items = st.session_state.get("last_results", [])
    if items:
        render_results(items)
    else:
        if run_search:
            st.info("No results.")

    # Handle deferred "Open" / "Reveal" AFTER rendering (so rerun doesn't clear the gallery)
    open_idx = st.session_state.pop("open_request", None)
    if open_idx is not None:
        try:
            target = st.session_state["last_results"][open_idx]["path"]
            if platform.system() == "Darwin":
                subprocess.call(["open", target])  # open in default app
                # To reveal in Finder instead of opening, use: ["open", "-R", target]
            elif platform.system() == "Windows":
                os.startfile(target)  # type: ignore[attr-defined]
            else:
                subprocess.call(["xdg-open", target])
            st.toast(f"Opening: {target}")
        except Exception as e:
            st.warning(f"Could not open file: {e}")

    reveal_idx = st.session_state.pop("reveal_request", None)
    if reveal_idx is not None:
        try:
            target = st.session_state["last_results"][reveal_idx]["path"]
            if platform.system() == "Darwin":
                subprocess.call(["open", "-R", target])  # reveal in Finder
            elif platform.system() == "Windows":
                # Best-effort reveal: open the folder
                folder = str(Path(target).parent)
                subprocess.call(["explorer", folder])
            else:
                # Linux: open folder containing file
                folder = str(Path(target).parent)
                subprocess.call(["xdg-open", folder])
            st.toast(f"Revealed: {target}")
        except Exception as e:
            st.warning(f"Could not reveal file: {e}")