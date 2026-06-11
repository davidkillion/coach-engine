"""
Discovery Coaching Engine
Location: /var/www/coach-engine/dev/main.py
Version: v2.00.0003

CHANGELOG:
v2.00.0003 - Removed ALL fallback philosophy logic. A primer MUST be sent on
             the first turn of every conversation (cached by hash thereafter).
             If a new conversation arrives with no primer and no cached match,
             the engine returns HTTP 422 — no silent generic-coach fallback.
             Deleted local financial_philosophy.md dependency entirely.
v2.00.0002 - Conversation history + primer-as-full-prompt + prompt caching.
v2.00.0001 - Philosophy caching by hash; Gemini 2.5-flash.
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
    handlers=[logging.FileHandler("logs/app.log"), logging.StreamHandler()]
)
logger = logging.getLogger("coach-engine")

log_buffer = []

def log(level: str, step: str, message: str, detail: str = ""):
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "level": level, "step": step, "message": message, "detail": detail
    }
    log_buffer.append(entry)
    if len(log_buffer) > 100:
        log_buffer.pop(0)
    getattr(logger, level.lower(), logger.info)(f"[{step}] {message} {detail}".strip())


# --- Philosophy Cache (by hash) — NO fallback ---
philosophy_cache: dict[str, str] = {}

def resolve_philosophy(philosophy: str, philosophy_hash: str) -> str | None:
    """
    Resolve the primer for this request.
    - If full philosophy + hash sent: cache and use it.
    - If only hash sent and we have it cached: use cache.
    - Otherwise: return None. NO fallback. Caller must reject the request.
    """
    if philosophy and philosophy_hash:
        if philosophy_hash not in philosophy_cache:
            philosophy_cache[philosophy_hash] = philosophy
            log("info", "CACHE", f"New philosophy cached — hash={philosophy_hash[:8]} len={len(philosophy)}")
        return philosophy_cache[philosophy_hash]

    if philosophy_hash and philosophy_hash in philosophy_cache:
        return philosophy_cache[philosophy_hash]

    # No primer available — explicit failure, no generic fallback.
    return None


# --- Conversation Store ---
conversations: dict[str, dict] = {}
MAX_TURNS = 20

def get_conversation(conversation_id: str, philosophy_hash: str) -> dict:
    conv = conversations.get(conversation_id)
    if conv is None or conv["hash"] != philosophy_hash:
        conversations[conversation_id] = {"hash": philosophy_hash, "messages": []}
        if conv is not None:
            log("info", "CONV", f"Primer changed — reset history for {conversation_id[:8]}")
        else:
            log("info", "CONV", f"New conversation {conversation_id[:8]}")
    return conversations[conversation_id]

def trim_history(conv: dict):
    msgs = conv["messages"]
    if len(msgs) > MAX_TURNS * 2:
        conv["messages"] = msgs[-(MAX_TURNS * 2):]


app = FastAPI(title="Coach Engine")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
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


# --- AI: Claude (history + prompt caching) ---
def run_claude(messages: list, philosophy: str, api_key: str) -> str:
    log("info", "AI:Claude", f"History depth: {len(messages)} msgs")
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=[{"type": "text", "text": philosophy, "cache_control": {"type": "ephemeral"}}],
        messages=messages,
    )
    text = message.content[0].text.strip()
    log("info", "AI:Claude", f"Response: {text[:80]}...")
    return text


# --- AI: Gemini (history) ---
async def run_gemini(messages: list, philosophy: str, api_key: str) -> str:
    log("info", "AI:Gemini", f"History depth: {len(messages)} msgs")
    contents = []
    for m in messages:
        role = "user" if m["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": m["content"]}]})
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}",
            json={
                "system_instruction": {"parts": [{"text": philosophy}]},
                "contents": contents,
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
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": "tts-1", "voice": "onyx", "input": text, "response_format": "wav"},
        )
        response.raise_for_status()
        data = response.content
        log("info", "TTS:OpenAI", f"Complete: {len(data)} bytes")
        return data


# --- TTS: ElevenLabs ---
async def elevenlabs_tts(text: str, api_key: str) -> bytes:
    voice_id = "pNInz6obpgDQGcFmaJgB"
    log("info", "TTS:ElevenLabs", f"Generating audio for {len(text)} chars")
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
            headers={"xi-api-key": api_key, "Content-Type": "application/json"},
            json={
                "text": text,
                "model_id": "eleven_turbo_v2_5",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}
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
    return {"engine": "Coach Engine", "status": "operational", "version": "v2.00.0003"}


@app.get("/logs")
def get_logs(n: int = 50):
    return JSONResponse(content=log_buffer[-n:])


@app.post("/upload-audio")
async def handle_audio_coaching(
    file:             UploadFile = File(...),
    ai:               str = Form(default="claude"),
    tts:              str = Form(default="openai"),
    conversation_id:  str = Form(default="default"),
    philosophy:       str = Form(default=""),
    philosophy_hash:  str = Form(default="")
):
    log("info", "REQUEST", f"ai={ai} tts={tts} conv={conversation_id[:8]} hash={philosophy_hash[:8] if philosophy_hash else 'none'}")

    allowed_extensions = ["mp3", "wav", "m4a", "webm"]
    file_extension = file.filename.split(".")[-1].lower()
    if file_extension not in allowed_extensions:
        log("error", "REQUEST", f"Unsupported format: {file_extension}")
        raise HTTPException(status_code=400, detail="Unsupported audio format.")

    # --- Primer required — no fallback ---
    philosophy_text = resolve_philosophy(philosophy, philosophy_hash)
    if philosophy_text is None:
        log("error", "PRIMER", "No primer available — rejecting request",
            f"hash={philosophy_hash[:8] if philosophy_hash else 'none'}")
        raise HTTPException(
            status_code=422,
            detail="No primer document available. The first turn of a conversation must include a philosophy document."
        )

    anthropic_key  = os.getenv("ANTHROPIC_API_KEY")
    openai_key     = os.getenv("OPENAI_API_KEY")
    gemini_key     = os.getenv("GEMINI_API_KEY")
    elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")

    try:
        audio_bytes = await file.read()
        conv = get_conversation(conversation_id, philosophy_hash)
        loop = asyncio.get_event_loop()

        # Step 1: Whisper STT
        filename  = f"user_voice.{file_extension}"
        user_text = await whisper_transcribe(audio_bytes, filename, openai_key)

        conv["messages"].append({"role": "user", "content": user_text})

        # Step 2: AI response with full history
        if ai == "gemini":
            coach_text = await run_gemini(conv["messages"], philosophy_text, gemini_key)
        else:
            coach_text = await loop.run_in_executor(None, run_claude, conv["messages"], philosophy_text, anthropic_key)

        conv["messages"].append({"role": "assistant", "content": coach_text})
        trim_history(conv)

        # Step 3: TTS
        if tts == "elevenlabs":
            audio_data = await elevenlabs_tts(coach_text, elevenlabs_key)
            media_type = "audio/mpeg"
        else:
            audio_data = await openai_tts(coach_text, openai_key)
            media_type = "audio/wav"

        log("info", "COMPLETE", f"Success — {len(audio_data)} bytes, history now {len(conv['messages'])} msgs")

        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type=media_type,
            headers={"X-Coach-Text": make_safe_header(coach_text)}
        )

    except HTTPException:
        raise
    except Exception as e:
        log("error", "ERROR", str(e), type(e).__name__)
        raise HTTPException(status_code=500, detail=f"Engine fault: {str(e)}")