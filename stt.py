from dotenv import load_dotenv
load_dotenv()

import os
import time
import logging
from deepgram import Deepgram

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

if not DEEPGRAM_API_KEY:
    raise ValueError("DEEPGRAM_API_KEY is missing.")

deepgram = Deepgram(DEEPGRAM_API_KEY)

async def transcribe_audio(file_path: str) -> tuple[str, float]:
    try:
        start = time.time()
        with open(file_path, 'rb') as audio:
            response = await deepgram.transcription.prerecorded(
                {
                    'buffer': audio,
                    'mimetype': 'audio/wav'
                },
                {
                    'punctuate': True,
                    'language': 'en'
                }
            )
        transcript = response['results']['channels'][0]['alternatives'][0]['transcript']
        elapsed = time.time() - start
        logging.info(f"[STT] Transcription took {elapsed:.2f}s: {transcript}")
        return transcript or "No speech detected.", elapsed
    except Exception:
        logging.exception("[STT] Deepgram transcription failed")
        return "Transcription failed.", 0.0
