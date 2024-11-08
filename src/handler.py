"""Example handler file for RunPod AI inference."""

import runpod
import torch
import re
import logging
from unsloth import FastLanguageModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_assistant_content(text):
    """Extract assistant's response from the conversation format."""
    pattern = r'<\|start_header_id\|>assistant<\|end_header_id\|>\n(.*?)<\|eot_id\|>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ''

def initialize_model():
    """Initialize the model with error handling."""
    try:
        max_seq_length = 8192
        dtype = None
        load_in_4bit = True
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="/cache/huggingface/hub/tekkilma-24000",
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            local_files_only=True,
        )
        FastLanguageModel.for_inference(model)
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        raise

# Initialize model and tokenizer globally
try:
    MODEL, TOKENIZER = initialize_model()
    logger.info("Model initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize model globally: {str(e)}")
    raise

def run_inference(system_prompt, sentence):
    """Run model inference with error handling."""
    try:
        # Validate inputs
        if not system_prompt or not sentence:
            raise ValueError("Both system_prompt and sentence must be provided")

        # Prepare messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": sentence}
        ]

        # Tokenize input
        inputs = TOKENIZER.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")

        # Generate response
        outputs = MODEL.generate(
            input_ids=inputs,
            max_new_tokens=8192,
            temperature=0.01,
            use_cache=True
        )

        # Decode and extract response
        raw_text = TOKENIZER.batch_decode(outputs)[0]
        #result = extract_assistant_content(raw_text)
        
        return {"success": True, "output": raw_text}

    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        return {"success": False, "error": str(e)}

def handler(event):
    """Handle incoming RunPod serverless requests."""
    try:
        # Validate event input
        if not isinstance(event, dict) or "input" not in event:
            raise ValueError("Invalid event format")

        job_input = event["input"]
        
        # Validate required fields
        system_prompt = job_input.get("system_prompt")
        sentence = job_input.get("sentence")
        
        if not system_prompt or not sentence:
            raise ValueError("Missing required fields: system_prompt and sentence")

        # Run inference
        result = run_inference(system_prompt, sentence)
        
        if result["success"]:
            return {"statusCode": 200, "output": result["output"]}
        else:
            return {"statusCode": 500, "error": result["error"]}

    except Exception as e:
        logger.error(f"Handler error: {str(e)}")
        return {"statusCode": 500, "error": str(e)}

# Start the RunPod serverless handler
if __name__ == "__main__":
    logger.info("Starting RunPod Serverless API")
    runpod.serverless.start({"handler": handler})