
# Windanam but very fast

```bash
docker build --platform linux/amd64 --tag yayasy/henyo-windanam:latest .
```

then push:

```bash
docker push yayasy/henyo-windanam:latest
```

Ensure that you have Docker installed and properly set up before running the docker build commands. Once built, you can deploy this serverless worker in your desired environment with confidence that it will automatically scale based on demand.

# Test Inputs

The following inputs can be used for testing the model:

```bash
curl --request POST \                      
     --url https://api.runpod.ai/v2/f1y2pncaxioyu8/runsync \
     --header "accept: application/json" \
     --header "authorization: L5P64411WDOWJVEDYOX75103OZBHNHZQE0QW7RG9" \
     --header "content-type: application/json" \
     --data '{
  "input": {
    "audio": "https://github.com/cawoylel/Segmentation/raw/main/sample/SEGMENTED_Soundcloud_00699f7a-1369-425f-bac0-342dac55aa90_79_9975fbb0-b97c-11ee-85a4-42010a800004.wav",
    "chunk_length": 16,
    "batch_size": 24,
    "language": "ha",
    "model": "cawoylel/windanam_whisper-medium"
  }
}
```

## Acknowledgments

- This tool is powered by Hugging Face's ASR models, primarily Whisper by OpenAI.
- Optimizations are developed by [Vaibhavs10/insanely-fast-whisper](https://github.com/Vaibhavs10/insanely-fast-whisper).
