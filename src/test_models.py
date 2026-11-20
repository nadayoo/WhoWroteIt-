"""test_models.py - which Gemini model answers fast for you?   python src\\test_models.py"""
import os, time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
from google import genai
from google.genai import types

CANDIDATES = ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-flash-lite-latest",
              "gemini-3.5-flash", "gemini-3.7-flash", "gemini-3.8-flash"]
c = genai.Client(api_key=os.environ["GEMINI_API_KEY"], http_options=types.HttpOptions(timeout=30_000))
for m in CANDIDATES:
    t = time.time()
    try:
        r = c.models.generate_content(model=m, contents="Reply with the word ok.",
                                      config=types.GenerateContentConfig(temperature=0))
        print(f"OK    {m:28s} {time.time() - t:5.1f}s  -> {r.text.strip()[:20]!r}", flush=True)
    except Exception as e:
        print(f"FAIL  {m:28s} {time.time() - t:5.1f}s  -> {type(e).__name__}: {str(e)[:90]}", flush=True)