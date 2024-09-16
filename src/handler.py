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

def decode_base64_audio(base64_audio, codec = "opus"):
    """Helper function to decode base64 audio."""
    audio_bytes = base64.b64decode(base64_audio)
    opus_data = BytesIO(audio_bytes)
    audiosegment = AudioSegment.from_file(opus_data, codec=codec)

    if audiosegment.channels==2:
      audiosegment = audiosegment.set_channels(1)

    samples = audiosegment.get_array_of_samples()
    sample_all = librosa.util.buf_to_float(samples,n_bytes=2,
                                      dtype=np.float32)
    if audiosegment.frame_rate != 16000:
        sample_all = librosa.resample(sample_all, orig_sr=audiosegment.frame_rate, target_sr=16000)
    return sample_all

def run_whisper_inference(audio_input, chunk_length, batch_size, language, task, model):
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
        generate_kwargs={"task": task, "language": language},
        return_timestamps=True,
    )

    return outputs["text"]


def handler(job):
    job_input = job['input']
    chunk_length = job_input["chunk_length"] if 'chunk_length' in job_input else 16
    batch_size = job_input["batch_size"] if 'batch_size' in job_input else 24
    language = job_input["language"] if "language" in job_input else "ha"
    task = job_input["task"] if "task" in job_input else "transcribe"
    model = job_input["model"] if "model" in job_input else "cawoylel/mawdo-windanam-3000"

    audio_input = None

    # Handle audio and transcription/translation

    if "audio_url" in job_input:
        audio_input = download_file(job_input["audio_url"], 'downloaded_audio.wav')
        
    elif "audio_base64" in job_input:
        clean_base64_string = clean_base64(job_input["audio_base64"])
        audio_input = download_recording(clean_base64_string, 'downloaded_audio.wav')

    if audio_input is not None:
        result = run_whisper_inference(audio_input, chunk_length, batch_size, language, task, model)
        if os.path.exists(audio_input):
                os.remove(audio_input)

        return result
        

runpod.serverless.start({"handler": handler})
