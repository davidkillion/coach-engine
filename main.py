"""
Discovery Coaching Engine
Location: /var/www/coach-engine/dev/main.py
Version: v2.00.0001

CHANGELOG:
v2.00.0001 - New versioning scheme. Added philosophy caching:
             accepts philosophy, philosophy_hash, conversation_id params.
             In-memory cache keyed by hash — only updates when hash changes.
             Falls back to local financial_philosophy.md if no doc sent.
             Fixed Gemini model to gemini-2.5-flash.
"""

import os
import io
import asyncio
import httpx
import anthropic
import logging
from datetime import datetime, timezone
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from dotenv import load_dotenv

load_dotenv()

# --- Logging Setup ---
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("coach-engine")

log_buffer = []

def log(level: str, step: str, message: str, detail: str = ""):
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "step": step,
        "message": message,
        "detail": detail
    }
    log_buffer.append(entry)
    if len(log_buffer) > 100:
        log_buffer.pop(0)
    getattr(logger, level.lower(), logger.info)(f"[{step}] {message} {detail}".strip())


# --- Philosophy Cache ---
# { hash: philosophy_text }
philosophy_cache: dict[str, str] = {}

def get_philosophy(philosophy: str, philosophy_hash: str) -> str:
    """
    Use provided philosophy doc if given.
    Cache by hash so we don't re-store identical docs.
    Fall back to local file if nothing provided.
    """
    if philosophy and philosophy_hash:
        if philosophy_hash not in philosophy_cache:
            philosophy_cache[philosophy_hash] = philosophy
            log("info", "CACHE", f"New philosophy cached — hash={philosophy_hash[:8]}... len={len(philosophy)}")
        else:
            log("info", "CACHE", f"Cache hit — hash={philosophy_hash[:8]}...")
        return philosophy_cache[philosophy_hash]

    # Fallback to local file
    local = "financial_philosophy.md"
    if os.path.exists(local):
        with open(local, "r", encoding="utf-8") as f:
            return f.read()
    return "You are a direct, no-fluff business and financial coach."


app = FastAPI(title="Coach Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def make_safe_header(text: str) -> str:
    return text.replace("\r", " ").replace("\n", " ").encode("ascii", errors="replace").decode("ascii")


# --- STT ---
async def whisper_transcribe(audio_bytes: bytes, filename: str, api_key: str) -> str:
    log("info", "STT", f"Sending {len(audio_bytes)} bytes to Whisper")
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": (filename, audio_bytes)},
            data={"model": "whisper-1"},
        )
        response.raise_for_status()
        text = response.json()["text"].strip()
        log("info", "STT", f"Transcript: {text}")
        return text


# --- AI: Claude ---
def run_claude(user_text: str, philosophy: str, api_key: str) -> str:
    log("info", "AI:Claude", f"Sending: {user_text[:80]}...")
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=f"""You are David Killion's coaching voice. Respond based on this philosophy:

{philosophy}

Voice: direct, grounded, honest over comfortable. Action over sympathy.
Empathy is brief — redirect to capability and next steps.
Never reinforce victimhood. Keep responses under 150 words.""",
        messages=[{"role": "user", "content": user_text}],
    )
    text = message.content[0].text.strip()
    log("info", "AI:Claude", f"Response: {text[:80]}...")
    return text


# --- AI: Gemini ---
async def run_gemini(user_text: str, philosophy: str, api_key: str) -> str:
    log("info", "AI:Gemini", f"Sending: {user_text[:80]}...")
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}",
            json={
                "system_instruction": {
                    "parts": [{"text": f"""You are David Killion's coaching voice. Respond based on this philosophy:

{philosophy}

Voice: direct, grounded, honest over comfortable. Action over sympathy.
Empathy is brief — redirect to capability and next steps.
Never reinforce victimhood. Keep responses under 150 words."""}]
                },
                "contents": [{"parts": [{"text": user_text}]}],
                "generationConfig": {"temperature": 0.7}
            }
        )
        if response.status_code != 200:
            log("error", "AI:Gemini", f"HTTP {response.status_code}", response.text[:300])
            response.raise_for_status()
        text = response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        log("info", "AI:Gemini", f"Response: {text[:80]}...")
        return text


# --- TTS: OpenAI ---
async def openai_tts(text: str, api_key: str) -> bytes:
    log("info", "TTS:OpenAI", f"Generating audio for {len(text)} chars")
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "https://api.openai.com/v1/audio/speech",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "tts-1",
                "voice": "onyx",
                "input": text,
                "response_format": "wav",
            },
        )
        response.raise_for_status()
        data = response.content
        log("info", "TTS:OpenAI", f"Complete: {len(data)} bytes")
        return data


# --- TTS: ElevenLabs ---
async def elevenlabs_tts(text: str, api_key: str) -> bytes:
    voice_id = "pNInz6obpgDQGcFmaJgB"  # Adam
    log("info", "TTS:ElevenLabs", f"Generating audio for {len(text)} chars")
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
            headers={
                "xi-api-key": api_key,
                "Content-Type": "application/json",
            },
            json={
                "text": text,
                "model_id": "eleven_turbo_v2_5",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75
                }
            },
        )
        if response.status_code != 200:
            log("error", "TTS:ElevenLabs", f"HTTP {response.status_code}", response.text[:300])
            response.raise_for_status()
        data = response.content
        log("info", "TTS:ElevenLabs", f"Complete: {len(data)} bytes")
        return data


@app.get("/")
def read_root():
    return {"engine": "Coach Engine", "status": "operational", "version": "v2.00.0001"}


@app.get("/logs")
def get_logs(n: int = 50):
    return JSONResponse(content=log_buffer[-n:])


@app.post("/upload-audio")
async def handle_audio_coaching(
    file:             UploadFile = File(...),
    ai:               str = Form(default="claude"),
    tts:              str = Form(default="openai"),
    conversation_id:  str = Form(default=""),
    philosophy:       str = Form(default=""),
    philosophy_hash:  str = Form(default="")
):
    log("info", "REQUEST", f"ai={ai} tts={tts} conv={conversation_id[:8] if conversation_id else 'none'} hash={philosophy_hash[:8] if philosophy_hash else 'none'}")

    allowed_extensions = ["mp3", "wav", "m4a", "webm"]
    file_extension = file.filename.split(".")[-1].lower()
    if file_extension not in allowed_extensions:
        log("error", "REQUEST", f"Unsupported format: {file_extension}")
        raise HTTPException(status_code=400, detail="Unsupported audio format.")

    anthropic_key  = os.getenv("ANTHROPIC_API_KEY")
    openai_key     = os.getenv("OPENAI_API_KEY")
    gemini_key     = os.getenv("GEMINI_API_KEY")
    elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")

    try:
        audio_bytes    = await file.read()
        active_philosophy = get_philosophy(philosophy, philosophy_hash)
        loop           = asyncio.get_event_loop()

        # Step 1: Whisper STT
        filename  = f"user_voice.{file_extension}"
        user_text = await whisper_transcribe(audio_bytes, filename, openai_key)

        # Step 2: AI response
        if ai == "gemini":
            coach_text = await run_gemini(user_text, active_philosophy, gemini_key)
        else:
            coach_text = await loop.run_in_executor(None, run_claude, user_text, active_philosophy, anthropic_key)

        # Step 3: TTS
        if tts == "elevenlabs":
            audio_data = await elevenlabs_tts(coach_text, elevenlabs_key)
            media_type = "audio/mpeg"
        else:
            audio_data = await openai_tts(coach_text, openai_key)
            media_type = "audio/wav"

        log("info", "COMPLETE", f"Pipeline success — {len(audio_data)} bytes returned")

        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type=media_type,
            headers={"X-Coach-Text": make_safe_header(coach_text)}
        )

    except Exception as e:
        log("error", "ERROR", str(e), type(e).__name__)
        raise HTTPException(status_code=500, detail=f"Engine fault: {str(e)}")