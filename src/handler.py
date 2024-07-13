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


def download_file(url, local_filename):
    """Helper function to download a file from a URL."""
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename


def run_whisper_inference(audio_path, chunk_length, batch_size, language, task, model):
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
        audio_path,
        chunk_length_s=chunk_length,
        batch_size=batch_size,
        generate_kwargs={"task": task, "language": language},
        return_timestamps=True,
    )

    return outputs["text"]


def handler(job):
    job_input = job['input']
    audio_url = job_input["audio"]
    chunk_length = job_input["chunk_length"]
    batch_size = job_input["batch_size"]
    language = job_input["language"] if "language" in job_input else None
    task = job_input["task"] if "task" in job_input else "transcribe"
    model = job_input["model"] if "model" in job_input else "openai/whisper-large-v3"

    if audio_url:
        # Download the audio file
        # TODO: use a unique filename id and supports youtube links.
        audio_file_path = download_file(audio_url, 'downloaded_audio.wav')

        # Run Whisper model inference
        result = run_whisper_inference(
            audio_file_path, chunk_length, batch_size, language, task, model)
        # Cleanup: Remove the downloaded file
        os.remove(audio_file_path)

        return result
    else:
        return "No audio URL provided."


runpod.serverless.start({"handler": handler})
