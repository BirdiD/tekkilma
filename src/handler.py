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
import json
from anthropic import AnthropicVertex, Anthropic

def extract_final_translation(text):
    last_number = re.findall(r'\d+\)', text)[-1]
    final_part = text.split(last_number)[-1].strip()
    final_translation = re.sub(r'^[:\s]+', '', final_part).split(":")[-1]
    return final_translation
    
class Translator:
    """
    Use Claude API through GCP or directly with Anthropic to translate a sentence
    Perform the translation GCP or Anthropic and if it fails (server overloaded), try with the other solution
    """

    def __init__(self):
        self.gcp_project_id = os.getenv("GCP_PROJECT_ID")
        self.gcp_location = os.getenv("GCP_LOCATION")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self._credentials_file_path = "/tmp/gcp_credentials.json"

        if not os.path.exists(self._credentials_file_path):
            gcp_service_account_json = os.getenv("GCP_CREDENTIALS")  
            if gcp_service_account_json:
                credentials_dict = json.loads(gcp_service_account_json)
                
                with open(self._credentials_file_path, 'w') as f:
                    json.dump(credentials_dict, f)
                
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self._credentials_file_path
        
        self.vertex_client = AnthropicVertex(
            region=self.gcp_location, project_id=self.gcp_project_id
        )
        self.anthropic_client = Anthropic(api_key=self.anthropic_api_key)

    def translate(self, sentence, source_lan, target_lan):
        result = self._translate_with_anthropic(sentence, source_lan, target_lan)
        if result:
            return result
        result = self._translate_with_vertex(sentence, source_lan, target_lan)
        if result:
            return result
        return "Mi ronkii firtude"

    def _translate_with_vertex(self, sentence, source_lan, target_lan):
        prompt = self._create_prompt(sentence, source_lan, target_lan)

        try:
            message = self.vertex_client.messages.create(
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
                model="claude-3-5-sonnet@20240620",
                temperature=0.01,
            )
            return message.content[0].text
        except Exception as e:
            print(f"AnthropicVertex translation failed: {e}")
            return

    def _translate_with_anthropic(self, sentence, source_lan, target_lan):
        prompt = self._create_prompt(sentence, source_lan, target_lan)

        try:
            message = self.anthropic_client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=2048,
                messages=[{"role": "user", "content": f"{prompt}"}],
                temperature=0.01,
            )
            return message.content[0].text
        except Exception as e:
            print(f"Anthropic translation failed: {e}")
            return

    def _create_prompt(self, sentence, source_lan, target_lan):
        return f"""You are a professional translator with expert-level fluency in Fula, English, and French. Your task is to provide a precise and accurate translation of the given sentence from {source_lan} to {target_lan}.

        Please adhere to the following guidelines:
        1. Translate the sentence faithfully, preserving its original meaning, tone, and nuance.
        2. Pay attention to idiomatic expressions and cultural context in both the source and target languages.
        3. Maintain the original sentence structure when possible, but prioritize natural expression in the target language.
        4. If the source sentence contains specialized terminology, translate it accurately using the appropriate terms in the target language.
        5. Ensure proper grammar, spelling, and punctuation in the target language.
        6. Do not add any explanations, notes, or additional text to your translation.

        Before translating, follow the following steps;
        1) Explain what means each word/expression in the text in {source_lan}.
        2) Propose a reformulation of the text in the {source_lan} language without additional content while keeping all the meaning of the original text.
        3) Then translate the whole text in {target_lan} language using the results of step 1 and 2.

        Sentence to translate from {source_lan} to {target_lan}: {sentence}
        """

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

    translator = Translator()
    job_input = job['input']
    chunk_length = job_input["chunk_length"]
    batch_size = job_input["batch_size"]
    language = job_input["language"] if "language" in job_input else None
    task = job_input["task"] if "task" in job_input else "transcribe"
    model = job_input["model"] if "model" in job_input else "openai/whisper-large-v3"
    action = job_input["action"]

    audio_input = None
    text_input = None
    result = None

    # Handle text translation
    if 'text' in job_input:
        text_input = job_input["text"]
        source_language = job_input["source_language"]
        target_language = job_input["target_language"]
        translation = translator.translate(text_input, source_language, target_language)
        return extract_final_translation(translation)

    # Handle audio and transcription/translation

    if "audio_url" in job_input:
        audio_input = download_file(job_input["audio_url"], 'downloaded_audio.wav')
        
    elif "audio_base64" in job_input:
        #audio_input = decode_base64_audio(job_input["audio_base64"])
        clean_base64_string = clean_base64(job_input["audio_base64"])
        audio_input = download_recording(clean_base64_string, 'downloaded_audio.wav')

    if audio_input is not None:
        result = run_whisper_inference(audio_input, chunk_length, batch_size, language, task, model)
        if os.path.exists(audio_input):
                os.remove(audio_input)

        if action == 'translate':
            source_language = job_input["source_language"] if job_input["source_language"] else "Fula"
            target_language = job_input["target_language"]
            translation = translator.translate(result, source_language, target_language)
            return extract_final_translation(translation)

        else:
            return result

runpod.serverless.start({"handler": handler})
