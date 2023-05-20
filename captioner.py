import asyncio
import concurrent.futures
import logging
import os
import time
from typing import List, Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from transformers import pipeline

LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")
MAX_URLS = int(os.getenv("MAX_URLS", 5))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 200))
MODEL = os.getenv("MODEL", "../models/Salesforce/blip-image-captioning-large")

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


app = FastAPI()


executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
captioner = None  # Placeholder for the captioner pipeline
is_initialized = asyncio.Event()  # Event to track initialization status
lock = asyncio.Lock()


def load_model():
    global captioner
    logger.info("Loading model...")
    # simpler model: "ydshieh/vit-gpt2-coco-en"
    captioner = pipeline(
        "image-to-text",
        model=MODEL,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    logger.info("Done loading model.")
    is_initialized.set()


class Image(BaseModel):
    url: Union[HttpUrl, List[HttpUrl]]  # url can be a string or a list of strings


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(asyncio.to_thread(load_model))


@app.get("/")
async def root():
    return {"message": "Hello World"}


# the image url is passed in as a "url" tag in the json body
@app.post("/caption/")
async def create_caption(image: Image):
    if isinstance(image.url, list) and len(image.url) > MAX_URLS:
        logger.debug(
            f"Request with more than {MAX_URLS} URLs received. Refusing the request."
        )

        raise HTTPException(
            status_code=400,
            detail=f"Maximum of {MAX_URLS} URLs can be processed at once",
        )
    async with lock:
        await is_initialized.wait()  # Wait until initialization is completed
        loop = asyncio.get_running_loop()

        start_time = time.time()
        # get the image url from the json body
        image_url = image.url
        try:
            # caption = await loop.run_in_executor(executor, captioner, *(image_url,))
            caption = await loop.run_in_executor(executor, captioner, image_url)
        except Exception as e:
            logger.error("Error during caption generation: %s", str(e))
            raise HTTPException(
                status_code=500,
                detail="An error occurred during caption generation. Please try again later.",
            )
        end_time = time.time()
        duration = end_time - start_time
        logger.debug("Captioning completed. Time taken: %s seconds.", duration)

        return {"caption": caption, "duration": duration}


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
