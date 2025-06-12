import os
import asyncio
import logging
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from stt import transcribe_audio
from tts import play_tts_stream
from llm import ask_gpt
from llm_gemini import ask_gemini

# === Load env vars ===
load_dotenv()

# === Setup Flask + Logging ===
app = Flask(__name__)
CORS(app)

logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

# === LLM Selector ===
def get_llm_function(llm_name: str):
    llm_name = llm_name.lower().strip()
    return ask_gpt if llm_name == "openai" else ask_gemini

# === /api/text ===
@app.route("/api/text", methods=["POST"])
def handle_text():
    data = request.json
    prompt = data.get("prompt", "")
    llm_name = data.get("llm", "openai")
    llm_func = get_llm_function(llm_name)

    logging.info(f"[TEXT] Received: {prompt} | LLM: {llm_name}")

    try:
        start = time.perf_counter()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        response, llm_time = loop.run_until_complete(llm_func(prompt))
        tts_time = play_tts_stream(response)
        total_time = time.perf_counter() - start

        return jsonify({
            "response": response,
            "timing": {
                "stt": 0.0,
                "llm": round(llm_time, 2),
                "tts": round(tts_time, 2),
                "total": round(total_time, 2)
            }
        })

    except Exception as e:
        logging.exception("[TEXT] Error")
        return jsonify({"error": str(e)}), 500

# === /api/audio ===
@app.route("/api/audio", methods=["POST"])
def handle_audio():
    audio_file = request.files.get("audio")
    llm_name = request.form.get("llm", "openai")
    llm_func = get_llm_function(llm_name)

    audio_path = "temp.wav"
    audio_file.save(audio_path)
    logging.info(f"[AUDIO] Received audio | LLM: {llm_name}")

    try:
        start = time.perf_counter()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        transcript, stt_time = loop.run_until_complete(transcribe_audio(audio_path))
        response, llm_time = loop.run_until_complete(llm_func(transcript))
        tts_time = play_tts_stream(response)
        total_time = time.perf_counter() - start

        return jsonify({
            "transcript": transcript,
            "response": response,
            "timing": {
                "stt": round(stt_time, 2),
                "llm": round(llm_time, 2),
                "tts": round(tts_time, 2),
                "total": round(total_time, 2)
            }
        })

    except Exception as e:
        logging.exception("[AUDIO] Error")
        return jsonify({"error": str(e)}), 500

# === Run Server ===
if __name__ == "__main__":
    logging.info("🚀 Starting Flask server...")
    app.run(debug=True)
