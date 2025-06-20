import os
import asyncio
import json
import websockets
import sounddevice as sd
from dotenv import load_dotenv
import pyaudio
import requests
import httpx
import logging
import wave
from io import BytesIO

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

# --- Config ---
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
DEEPGRAM_URL = "wss://api.deepgram.com/v1/listen?punctuate=true&language=en"

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID = "EXAVITQu4vr4xnSDxMaL"
MODEL_ID = "eleven_multilingual_v2"

# --- Lock to prevent response overlap ---
response_lock = asyncio.Lock()

# --- Helpers ---
def pcm_to_wav(pcm_bytes: bytes, sample_rate=16000, channels=1):
    buf = BytesIO()
    wf = wave.open(buf, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(2)
    wf.setframerate(sample_rate)
    wf.writeframes(pcm_bytes)
    wf.close()
    return buf.getvalue()

async def ask_gpt_streaming(prompt: str):
    url = f"{AZURE_OPENAI_ENDPOINT}openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"
    headers = {"api-key": AZURE_OPENAI_API_KEY, "Content-Type": "application/json"}
    payload = {
        "messages": [
            {"role": "system", "content": "You are a concise voice assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 300,
        "stream": True
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    yield line[6:]

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
            "stability": 0.4,
            "similarity_boost": 0.6
        }
    }

    r = requests.post(url, headers=headers, json=payload, stream=True)
    r.raise_for_status()
    pcm = b"".join(r.iter_content(1024))
    logging.info(f"🔉 Received PCM chunk: {len(pcm)} bytes")
    return pcm_to_wav(pcm)

async def stream_tts_from_gpt(prompt: str):
    buffer = ""
    chunk_limit = 80

    logging.info("🧠 GPT → TTS streaming start")
    async for token in ask_gpt_streaming(prompt):
        piece = json.loads(token)["choices"][0]["delta"].get("content", "")
        if not piece:
            continue
        buffer += piece
        logging.info(f"💬 GPT: {piece.strip()}")

        if buffer.endswith(".") or len(buffer) > chunk_limit:
            wav = stream_tts_chunk(buffer)
            yield wav
            buffer = ""

    if buffer.strip():
        wav = stream_tts_chunk(buffer)
        yield wav
    logging.info("✅ GPT → TTS streaming done")

async def deepgram_mic_stream():
    logging.info("🎧 Connecting to Deepgram...")
    RATE = 16000
    CHUNK = 1024

    async with websockets.connect(DEEPGRAM_URL, extra_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"}) as ws:
        stop_event = asyncio.Event()

        loop = asyncio.get_event_loop()

        def callback(indata, frames, time, status):
            if not ws.closed:
                loop.call_soon_threadsafe(asyncio.create_task, ws.send(indata.tobytes()))

        async def send_audio():
            try:
                with sd.InputStream(samplerate=RATE, channels=1, dtype='int16', blocksize=CHUNK, callback=callback):
                    await stop_event.wait()
            except Exception as e:
                logging.exception("❌ Error in audio input stream")

        async def receive_transcript():
            try:
                async for msg in ws:
                    data = json.loads(msg)
                    transcript = data.get("channel", {}).get("alternatives", [{}])[0].get("transcript", "")
                    if transcript:
                        logging.info(f"📝 You said: {transcript}")
                        async with response_lock:
                            async for wav_chunk in stream_tts_from_gpt(transcript):
                                # Play audio using PyAudio
                                play_audio(wav_chunk)
            except websockets.exceptions.ConnectionClosed:
                logging.info("🔌 WebSocket closed.")
                stop_event.set()

        await asyncio.gather(send_audio(), receive_transcript())

def play_audio(wav_data: bytes):
    wf = wave.open(BytesIO(wav_data), 'rb')
    p = pyaudio.PyAudio()
    stream = p.open(
        format=p.get_format_from_width(wf.getsampwidth()),
        channels=wf.getnchannels(),
        rate=wf.getframerate(),
        output=True
    )
    chunk = 1024
    data = wf.readframes(chunk)
    while data:
        stream.write(data)
        data = wf.readframes(chunk)

    stream.stop_stream()
    stream.close()
    p.terminate()

# --- Main ---
if __name__ == "__main__":
    try:
        asyncio.run(deepgram_mic_stream())
    except KeyboardInterrupt:
        logging.info("👋 Exiting cleanly.")
