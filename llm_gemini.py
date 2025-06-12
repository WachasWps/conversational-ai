import os
import time
import logging
import httpx

# Set your Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or "YOUR_API_KEY"
GEMINI_MODEL = "gemini-1.5-flash"  # From AI Studio

async def ask_gemini(prompt: str) -> tuple[str, float]:
    try:
        start = time.time()

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
        headers = { "Content-Type": "application/json" }
        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ]
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()

            data = response.json()
            output = data["candidates"][0]["content"]["parts"][0]["text"]
            elapsed = time.time() - start

            logging.info(f"[GEMINI] LLM took {elapsed:.2f}s: {output}")
            return output, elapsed

    except Exception:
        logging.exception("[GEMINI] Gemini API error")
        return "Gemini failed to respond.", 0.0
