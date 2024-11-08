import os
import torch
from unsloth import FastLanguageModel
import torch


import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_unsloth():
  max_seq_length = 8192
  dtype = None
  load_in_4bit = True
  model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Soyno/tekkilma-24000",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    )
  return model, tokenizer

def fetch_pretrained_model():
    max_retries = 3
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempting to fetch model {model_name}, attempt {attempt + 1}")
            model, tokenizer = load_unsloth()
            return model, tokenizer
        except OSError as err:
            if attempt < max_retries - 1:
                logger.warning(f"Error encountered: {err}. Retrying attempt {attempt + 1} of {max_retries}...")
            else:
                logger.error(f"Failed to fetch model after {max_retries} attempts. Error: {err}")
                raise


def get_model():
    logger.info(f"Fetching model and Tokenizers")
    model, tokenizer = fetch_pretrained_model()
    
    logger.info("Saving model to disk")

    model.save_pretrained("workspace/tekkilma-24000") # Local saving
    tokenizer.save_pretrained("workspace/tekkilma-24000")    
    
    return model, tokenizer

if __name__ == "__main__":
    if os.environ.get("HF_HOME") != "/cache/huggingface":
        logger.error(f"HF_HOME is set to {os.environ.get('HF_HOME')}")
        raise ValueError("HF_HOME must be set to /cache/huggingface")
    
    logger.info("Starting model caching process")

    try:
        get_model()
        logger.info("Model caching completed successfully")
    except Exception as e:
        logger.error(f"An error occurred during model caching: {str(e)}")
        raise