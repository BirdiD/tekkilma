""" Example handler file. """

import runpod
import requests
import os
import torch
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizerFast,
    WhisperForConditionalGeneration,
    pipeline
)
import librosa
import numpy as np
from io import BytesIO
from pydub import AudioSegment
import base64
import re
import uuid
#from werkzeug.datastructures import FileStorage
from flask import request, jsonify
from werkzeug.utils import secure_filename
import logging


def download_file(url, local_filename):
    """Helper function to download a file from a URL."""
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename

def download_recording(base64_audio, local_filename):
    """Download the recording in wav format"""
    decoded_audio = base64.b64decode(base64_audio)
    audio_bytes = BytesIO(decoded_audio)
    audiosegment = AudioSegment.from_file(audio_bytes)
    audiosegment.export(local_filename, format="wav")
    return local_filename

def clean_base64(base64_string):
    cleaned_string = re.sub(r'^data:application/octet-stream;base64,', '', base64_string)
    return cleaned_string



def run_whisper_inference(audio_input, chunk_length, batch_size, model):
    """Run Whisper model inference on the given audio file."""
    model_id = model
    torch_dtype = torch.float16
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_cache = "/cache/huggingface/hub"
    local_files_only = True
    # Load the model, tokenizer, and feature extractor
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        cache_dir=model_cache,
        local_files_only=local_files_only,
    ).to(device)
    tokenizer = WhisperTokenizerFast.from_pretrained(
        model_id, cache_dir=model_cache, local_files_only=local_files_only
    )
    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        model_id, cache_dir=model_cache, local_files_only=local_files_only
    )

    # Initialize the pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        model_kwargs={"use_flash_attention_2": True},
        torch_dtype=torch_dtype,
        device=device,
    )

    # Run the transcription
    outputs = pipe(
        audio_input,
        chunk_length_s=chunk_length,
        batch_size=batch_size,
        #generate_kwargs={"task": task},
        return_timestamps=True,
    )

    return outputs["text"]


def handler(event):
    try:
        audio_file = None
        job_input = {}
        chunk_length = 16
        batch_size = 24
        model = "cawoylel/mawdo-windanam-3000"

        audio_input = None

        # Determine if the event contains multipart form-data or JSON input
        if "audio_file" not in request.files:
            # Multipart form data case
            audio_file = request.files["audio_file"]
            filename = secure_filename(audio_file.filename)
            audio_input = f'{str(uuid.uuid4())}_{filename}'
            audio_file.save(audio_input)

            chunk_length = request.form.get("chunk_length") if request.form.get("chunk_length") else chunk_length
            batch_size = request.form.get("batch_size") if request.form.get("chunk_length") else batch_size
            model = request.form.get("model") if request.form.get("chunk_length") else model
        else:
            # Regular JSON input case
            job_input = event.get('input', {})
            chunk_length = int(job_input.get("chunk_length", chunk_length))
            batch_size = int(job_input.get("batch_size", batch_size))
            model = job_input.get("model", model)

            if 'audio_url' in job_input:
                audio_input = download_file(job_input["audio_url"], f'{str(uuid.uuid4())}.wav')
            elif 'audio_base64' in job_input:
                clean_base64_string = clean_base64(job_input["audio_base64"])
                audio_input = download_recording(clean_base64_string, f'{str(uuid.uuid4())}.wav')
            else:
                logging.error("No valid audio input provided.")
                return "No valid audio input provided."

        # Run the Whisper inference if audio input is available
        if audio_input is not None:
            result = run_whisper_inference(audio_input, chunk_length, batch_size, model)
            if os.path.exists(audio_input):
                os.remove(audio_input)
            return result
        else:
            logging.error("Failed to process audio input.")
            return "Failed to process audio input."

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return f"An internal error occurred: {str(e)}"

runpod.serverless.start({"handler": handler})