import os
import time
import logging
import httpx

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

headers = {
    "api-key": AZURE_OPENAI_API_KEY,
    "Content-Type": "application/json"
}

async def ask_gpt(prompt: str) -> tuple[str, float]:
    try:
        start = time.time()
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 300
        }

        url = f"{AZURE_OPENAI_ENDPOINT}openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"

        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()["choices"][0]["message"]["content"]
            elapsed = time.time() - start
            logging.info(f"[GPT] GPT took {elapsed:.2f}s: {result}")
            return result, elapsed
    except Exception:
        logging.exception("[GPT] GPT API error")
        return "GPT failed.", 0.0
