import asyncio
import logging
import os
import time
from io import BytesIO
from typing import Optional

import gradio as gr
import PIL
import requests
import torch
from fastapi import FastAPI, HTTPException
from pillow_heif import register_heif_opener
from pydantic import BaseModel, HttpUrl
from transformers import pipeline

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")
# MAX_URLS = int(os.getenv("MAX_URLS", 5))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 2000))
# https://huggingface.co/models?pipeline_tag=image-to-text&sort=likes
MODEL = os.getenv("MODEL", "Salesforce/blip-image-captioning-large")
# simpler model: "ydshieh/vit-gpt2-coco-en"


register_heif_opener()

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


app = FastAPI()


captioner = None  # Placeholder for the captioner pipeline
is_initialized = asyncio.Event()  # Event to track initialization status
lock = asyncio.Lock()


def load_model():
    global captioner
    logger.info("Loading model...")
    start_time = time.perf_counter()
    try:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        logger.info("Using device: %s", device)
        captioner = pipeline(
            "image-to-text",
            model=MODEL,
            max_new_tokens=MAX_NEW_TOKENS,
            device=device,
        )
    except Exception as e:
        logger.error("Error loading model: %s", str(e))
        raise e
    end_time = time.perf_counter()
    duration = end_time - start_time
    logger.info("Done, model loaded in %.2f seconds.", duration)
    is_initialized.set()


@app.on_event("startup")
async def startup_event():
    global app
    asyncio.create_task(asyncio.to_thread(load_model))
    # add gradio interface
    iface = gr.Interface(
        fn=captioner_gradapter,
        inputs=[gr.Image(type="pil"), gr.Textbox(lines=1, placeholder="Image URL")],
        outputs=["text"],
        allow_flagging="never",
        interpretation="default",
    )
    app = gr.mount_gradio_app(app, iface, path="/")


async def captioner_gradapter(image, url):
    await is_initialized.wait()
    async with lock:
        start_time = time.perf_counter()
        input = image if image else url
        result = await asyncio.to_thread(captioner, input)
        caption = result[0]["generated_text"]
        end_time = time.perf_counter()
        duration = end_time - start_time
        logger.info("Captioning took %.2f seconds", duration)
    return caption


class Image(BaseModel):
    url: Optional[HttpUrl] = None
    data: Optional[bytes] = None


# the image url is passed in as a "url" tag in the json body
@app.post("/caption/")
async def create_caption(image: Image):
    async with lock:
        await is_initialized.wait()  # Wait until initialization is completed
        start_time = time.perf_counter()
        # get the image url from the json body
        if image.url is not None:
            image = image.url
        elif image.data is not None:
            image = Image.open(BytesIO(image.data))
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid request. Please pass in a valid image URL or image data.",
            )
        logger.debug("Received request for image: %s", image)
        try:
            caption = await asyncio.to_thread(captioner, str(image))
        except Exception as e:
            logger.error("Error during caption generation: %s", str(e))
            raise HTTPException(
                status_code=500,
                detail="An error occurred during caption generation. Please try again later.",
            )
        end_time = time.perf_counter()
        duration = end_time - start_time
        logger.debug("Captioning completed. Time taken: %s seconds.", duration)

        return {"caption": caption[0]["generated_text"], "duration": duration}


# add liveness probe
@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


# add readiness probe
@app.get("/readyz")
async def readyz():
    if not is_initialized.is_set():
        raise HTTPException(status_code=503, detail="Initialization in progress")
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
