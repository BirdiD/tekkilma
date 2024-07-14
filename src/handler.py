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
import base64
import io
import librosa


def download_file(url, local_filename):
    """Helper function to download a file from a URL."""
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename


def decode_base64_audio(base64_audio):
    """Helper function to decode base64 audio."""
    audio_bytes = base64.b64decode(base64_audio)
    audio, sample_rate = librosa.load(io.BytesIO(audio_bytes))
    if sample_rate != 16000:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
    return audio

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
    chunk_length = job_input["chunk_length"]
    batch_size = job_input["batch_size"]
    language = job_input["language"] if "language" in job_input else None
    task = job_input["task"] if "task" in job_input else "transcribe"
    model = job_input["model"] if "model" in job_input else "openai/whisper-large-v3"

    audio_input = None
    if "audio_url" in job_input:
        audio_input = download_file(job_input["audio_url"], 'downloaded_audio.wav')
    elif "audio_base64" in job_input:
        audio_input = decode_base64_audio(job_input["audio_base64"])
    else:
        return "No audio input provided. Please provide either 'audio_url' or 'audio_base64'."

    result = run_whisper_inference(
            audio_input, chunk_length, batch_size, language, task, model)
    
    # Cleanup: Remove the downloaded file if it exists
    if os.path.exists(audio_input):
        os.remove(audio_input)

    return result

runpod.serverless.start({"handler": handler})
