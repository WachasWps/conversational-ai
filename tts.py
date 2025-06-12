import os
import time
import logging
import requests
import pyaudio

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

def play_tts_stream(text: str) -> float:
    try:
        start = time.time()

        voice_id = "EXAVITQu4vr4xnSDxMaL"  # Rachel
        model_id = "eleven_multilingual_v2"

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream?optimize_streaming_latency=0"  # ultra-low
        headers = {
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json"
        }
        payload = {
            "text": text,
            "model_id": model_id,
            "voice_settings": {
                "stability": 0.3,
                "similarity_boost": 0.7,
                "style": 0,
                "use_speaker_boost": True
            }
        }

        # Start audio stream
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=22050, output=True)

        with requests.post(url, headers=headers, json=payload, stream=True) as response:
            response.raise_for_status()
            # start playing audio chunks as they stream
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    stream.write(chunk)

        stream.stop_stream()
        stream.close()
        p.terminate()

        elapsed = time.time() - start
        logging.info(f"[TTS] ElevenLabs streaming took {elapsed:.2f}s")
        return elapsed

    except Exception:
        logging.exception("[TTS] Error")
        return 0.0
