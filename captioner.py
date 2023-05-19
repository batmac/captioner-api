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
MAX_WORKERS = int(os.getenv("MAX_WORKERS", 5))
MAX_URLS = int(os.getenv("MAX_URLS", 5))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 200))
MODEL = os.getenv("MODEL", "../models/Salesforce/blip-image-captioning-large")


logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


app = FastAPI()

logger.info("Loading model...")
# simpler model: "ydshieh/vit-gpt2-coco-en"
captioner = pipeline(
    "image-to-text",
    model=MODEL,
    max_new_tokens=MAX_NEW_TOKENS,
)
logger.info("Done loading model.")

# semaphore = asyncio.Semaphore(5)
executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS)


class Image(BaseModel):
    url: Union[HttpUrl, List[HttpUrl]]  # url can be a string or a list of strings


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
            status_code=400, detail="Maximum of 5 URLs can be processed at once"
        )
    # async with semaphore:
    loop = asyncio.get_running_loop()

    start_time = time.time()
    # get the image url from the json body
    image_url = image.url
    caption = await loop.run_in_executor(executor, captioner, image_url)
    end_time = time.time()
    duration = end_time - start_time
    logger.debug("Captioning completed. Time taken: %s seconds.", duration)

    return {"caption": caption, "duration": duration}
