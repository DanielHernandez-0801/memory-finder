from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
from functools import lru_cache
from config import BLIP_MODEL

@lru_cache(maxsize=1)
def _blip():
    processor = BlipProcessor.from_pretrained(BLIP_MODEL)
    model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL)
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    model.to(device)
    return processor, model, device

def caption_image(pil_image: Image.Image) -> str:
    processor, model, device = _blip()
    inputs = processor(images=pil_image, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=30)
    text = processor.decode(out[0], skip_special_tokens=True)
    return text
