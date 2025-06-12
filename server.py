import os
import asyncio
import logging
import json
import httpx
import requests
import websockets
from quart import Quart, websocket
from dotenv import load_dotenv
from collections import deque

load_dotenv()
app = Quart(__name__)

# ENV
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
AZURE_OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID = "EXAVITQu4vr4xnSDxMaL"
MODEL_ID = "eleven_multilingual_v2"

logging.basicConfig(level=logging.INFO)

# === Azure GPT Streaming ===
async def stream_gpt(prompt):
    url = f"{AZURE_OPENAI_ENDPOINT}openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"
    headers = {
        "api-key": AZURE_OPENAI_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 500,
        "stream": True
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    try:
                        yield json.loads(line[6:])["choices"][0]["delta"].get("content", "")
                    except:
                        continue

# === ElevenLabs TTS Streaming
def get_tts_audio(text_chunk: str) -> bytes:
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream?optimize_streaming_latency=0&output_format=mp3_44100"

    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "text": text_chunk,
        "model_id": MODEL_ID,
        "voice_settings": {
            "stability": 0.4,
            "similarity_boost": 0.7,
            "style": 0.0,
            "use_speaker_boost": True
        }
    }
    response = requests.post(url, headers=headers, json=payload, stream=True)
    return b"".join(response.iter_content(1024))

# === Deepgram Live Transcription Handler
@app.websocket("/ws/live")
async def live_conversation():
    logging.info("🌐 WebSocket connection started")
    ws = websocket._get_current_object()

    audio_queue = asyncio.Queue()
    gpt_trigger = asyncio.Event()

    # === Receive audio from frontend
    async def receive_audio():
        try:
            while True:
                message = await websocket.receive()
                if isinstance(message, bytes):
                    logging.debug("📥 Received audio")
                    await audio_queue.put(message)
        except Exception as e:
            logging.error(f"❌ receive_audio error: {e}")

    # === Stream audio to Deepgram
    async def transcribe_audio():
        uri = f"wss://api.deepgram.com/v1/listen?punctuate=true&language=en"
        headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}

        async with websockets.connect(uri, extra_headers=headers) as dg_ws:
            async def send_audio():
                while True:
                    chunk = await audio_queue.get()
                    await dg_ws.send(chunk)

            async def receive_transcript():
                async for msg in dg_ws:
                    data = json.loads(msg)
                    transcript = data.get("channel", {}).get("alternatives", [{}])[0].get("transcript", "")
                    if transcript:
                        logging.info(f"📝 Transcript: {transcript}")
                        gpt_trigger.transcript = transcript
                        gpt_trigger.set()

            await asyncio.gather(send_audio(), receive_transcript())

    # === Handle GPT + TTS Response Streaming
    async def respond_to_audio():
        while True:
            await gpt_trigger.wait()
            prompt = gpt_trigger.transcript
            gpt_trigger.clear()

            buffer = ""
            async for token in stream_gpt(prompt):
                buffer += token
                logging.info(f"💬 GPT: {token.strip()}")
                if buffer.endswith(".") or len(buffer) > 80:
                    audio = get_tts_audio(buffer)
                    try:
                        await ws.send(audio)
                    except Exception:
                        logging.warning("⚠️ Client disconnected while sending audio.")
                        return
                    buffer = ""

            if buffer:
                audio = get_tts_audio(buffer)
                try:
                    await ws.send(audio)
                except Exception:
                    logging.warning("⚠️ Client disconnected during final audio.")
                    return

    try:
        await asyncio.gather(receive_audio(), transcribe_audio(), respond_to_audio())
    except Exception as e:
        logging.error(f"❌ WebSocket error: {e}")
    finally:
        logging.info("👋 Connection closed.")

# === Start Server ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
