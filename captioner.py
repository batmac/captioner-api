from transformers import pipeline
import gradio as gr
import logging
import time
import os
import torch
from pillow_heif import register_heif_opener

register_heif_opener()


os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 2000))
# https://huggingface.co/models?pipeline_tag=image-to-text&sort=likes
MODEL = os.getenv("MODEL", "Salesforce/blip-image-captioning-base")
# simpler model: "ydshieh/vit-gpt2-coco-en"

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

captioner = None


def load_model() -> pipeline:
    logger.info("Loading model...")
    start_time = time.perf_counter()
    try:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        logger.info("Using device: %s", device)
        captioner = pipeline(
            "image-to-text",
            model=MODEL,
            max_new_tokens=MAX_NEW_TOKENS,
            device=device,
            torch_dtype=torch.float16,
        )
    except Exception as e:
        logger.error("Error loading model: %s", str(e))
        raise e
    end_time = time.perf_counter()
    duration = end_time - start_time
    logger.info("Done, model loaded in %.2f seconds.", duration)
    return captioner


def caption(image):
    global captioner
    start_time = time.perf_counter()
    result = captioner(image)
    c = result[0]["generated_text"]
    end_time = time.perf_counter()
    duration = end_time - start_time
    logger.info("Captioning took %.2f seconds", duration)
    return c


captioner = load_model()

iface = gr.Interface(
    fn=caption,
    inputs=[gr.Image(type="pil")],  # gr.Textbox(lines=1, placeholder="Image URL")],
    outputs=["text"],
    allow_flagging="never",
)

logger.info("Starting gradio interface...")
iface.launch(share=True)
