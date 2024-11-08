""" Example handler file. """

import runpod
import requests
import os
from unsloth import FastLanguageModel
import torch
import re

def extract_assistant_content(text):
    pattern = r'<\|start_header_id\|>assistant<\|end_header_id\|>\n(.*?)<\|eot_id\|>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ''


def run_inference(system_prompt, sentence):

    max_seq_length = 8192
    dtype = None
    load_in_4bit = True
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "workspace/tekkilma-24000",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        local_files_only = True,
        )

    FastLanguageModel.for_inference(model)

    messages_1 = [
                {"role": "system", "content": f"{system_prompt}"},
                {"role": "user", "content": f"{sentence}"}
            ]

    inputs = tokenizer.apply_chat_template(
                messages_1,
                tokenize = True,
                add_generation_prompt = True,
                return_tensors = "pt",
            ).to("cuda")
    
    outputs = model.generate(input_ids = inputs, max_new_tokens = 8192, temperature=0.01, use_cache = True)
    raw_text = tokenizer.batch_decode(outputs)[0]
    return raw_text

def handler(event):
    job_input = event['input']
    system_prompt = job_input.get("system_prompt")
    sentence = job_input.get("sentence")



    # Run the Whisper inference if audio input is available
    if sentence is not None:
        result = run_inference(system_prompt, sentence)
        return result
    else:
        ogging.error("Failed to process audio input.")
        return "Failed to process audio input."

runpod.serverless.start({"handler": handler})
