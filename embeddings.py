from PIL import Image
import torch
import open_clip
import numpy as np
from config import CLIP_MODEL, CLIP_PRETRAINED

_device = "cpu"
if torch.backends.mps.is_available():
    _device = "mps"
elif torch.cuda.is_available():
    _device = "cuda"

_model, _, _preprocess = open_clip.create_model_and_transforms(CLIP_MODEL, pretrained=CLIP_PRETRAINED, device=_device)
_tokenizer = open_clip.get_tokenizer(CLIP_MODEL)

def image_embedding(pil_image: Image.Image) -> np.ndarray:
    img = _preprocess(pil_image).unsqueeze(0).to(_device)
    with torch.no_grad():
        feat = _model.encode_image(img)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().astype("float32")[0]

def text_embedding(text: str) -> np.ndarray:
    toks = _tokenizer([text])
    with torch.no_grad():
        txt = _model.encode_text(toks.to(_device))
        txt = txt / txt.norm(dim=-1, keepdim=True)
    return txt.cpu().numpy().astype("float32")[0]
