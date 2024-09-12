import os
import torch
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizerFast,
    WhisperForConditionalGeneration,
    pipeline,
    AutoProcessor
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

token = os.environ.get('HUGGING_FACE_HUB_WRITE_TOKEN')

def fetch_pretrained_model(model_class, model_name, **kwargs):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempting to fetch model {model_name}, attempt {attempt + 1}")
            return model_class.from_pretrained(model_name, **kwargs)
        except OSError as err:
            if attempt < max_retries - 1:
                logger.warning(f"Error encountered: {err}. Retrying attempt {attempt + 1} of {max_retries}...")
            else:
                logger.error(f"Failed to fetch model after {max_retries} attempts. Error: {err}")
                raise

def get_pipeline(model, tokenizer, feature_extractor, torch_dtype, device):
    logger.info("Initializing pipeline")
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        model_kwargs={"use_flash_attention_2": True},
        torch_dtype=torch_dtype,
        device=device,
    )
    return pipe

def get_model(model_id, device, torch_dtype):
    logger.info(f"Fetching model: {model_id}")
    model = fetch_pretrained_model(
        WhisperForConditionalGeneration,
        model_id,
        torch_dtype=torch_dtype,
        revision='e27f768494e4a3bf69b7a68b5fcc701abc1449e2',
        token=token
    ).to(device)
    
    logger.info("Fetching processor")
    processor = AutoProcessor.from_pretrained(model_id, token=token)
    
    logger.info("Initializing pipeline")
    #tokenizer = WhisperTokenizerFast.from_pretrained(model_id)
    #feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)
    get_pipeline(model, processor.tokenizer, processor.feature_extractor, torch_dtype, device)
    
    return model, processor.tokenizer, processor.feature_extractor

if __name__ == "__main__":
    if os.environ.get("HF_HOME") != "/cache/huggingface":
        logger.error(f"HF_HOME is set to {os.environ.get('HF_HOME')}")
        raise ValueError("HF_HOME must be set to /cache/huggingface")
    
    logger.info("Starting model caching process")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    try:
        get_model("cawoylel/mawdo-windanam", device, torch.float16)
        logger.info("Model caching completed successfully")
    except Exception as e:
        logger.error(f"An error occurred during model caching: {str(e)}")
        raise