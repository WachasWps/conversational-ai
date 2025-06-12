import os
import asyncio
import json
import logging
import httpx
import requests
import pyaudio
from dotenv import load_dotenv

load_dotenv()

# === Azure OpenAI Config ===
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")

# === ElevenLabs Config ===
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID = "EXAVITQu4vr4xnSDxMaL"
MODEL_ID = "eleven_multilingual_v2"

# === Function: Stream GPT (OpenAI) via SSE ===
async def ask_gpt_streaming(prompt: str):
    url = f"{AZURE_OPENAI_ENDPOINT}openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"
    headers = {
        "api-key": AZURE_OPENAI_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 300,
        "stream": True
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    yield line[6:]

# === Function: Extract Token Text ===
def extract_text_from_token(json_str):
    try:
        data = json.loads(json_str)
        return data["choices"][0]["delta"].get("content", "")
    except Exception:
        return ""

# === Function: Convert Short Text Chunk to Audio Bytes ===
def stream_tts_chunk(text: str) -> bytes:
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream?optimize_streaming_latency=0&output_format=pcm_16000"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "text": text,
        "model_id": MODEL_ID,
        "voice_settings": {
            "stability": 0.3,
            "similarity_boost": 0.7,
            "style": 0.0,
            "use_speaker_boost": True
        }
    }
    response = requests.post(url, headers=headers, json=payload, stream=True)
    response.raise_for_status()
    return b"".join(chunk for chunk in response.iter_content(chunk_size=1024))

# === Function: Stream GPT Tokens to TTS Playback ===
async def stream_tts_from_gpt(prompt: str):
    text_buffer = ""
    chunk_limit = 40  # characters or more before sending to TTS

    p = pyaudio.PyAudio()
    stream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=16000,
    output=True
)

    async for token in ask_gpt_streaming(prompt):
        text_piece = extract_text_from_token(token)
        if not text_piece:
            continue
        text_buffer += text_piece

        if len(text_buffer) >= chunk_limit and text_buffer.strip().endswith("."):
            audio = stream_tts_chunk(text_buffer)
            stream.write(audio)
            text_buffer = ""

    # Final flush
    if text_buffer.strip():
        audio = stream_tts_chunk(text_buffer)
        stream.write(audio)

    stream.stop_stream()
    stream.close()
    p.terminate()

# === Main Execution ===
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    user_prompt = input("You: ")
    asyncio.run(stream_tts_from_gpt(user_prompt))
