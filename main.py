"""
Discovery Coaching Engine
Location: /var/www/coach-engine/dev/main.py
Version: v2.00.0014

CHANGELOG:
v2.00.0014 - Text-out path (tts="none") for typed/chat consumers:
             * Both /coach-text and /upload-audio now accept tts="none".
               When set, the engine runs STT (audio path only) + AI, then
               returns the coach text as JSON and SKIPS TTS entirely.
               No OpenAI/ElevenLabs TTS call = no audio latency, no TTS cost.
             * Response shape for tts="none":
                 {"coach_text": "...", "conversation_id": "..."}
             * Enables the tri-mode RestartWorks frontend (/test2): typing and
               voice-typing use tts="none" (fast text reply); conversation mode
               keeps the existing streaming-audio path unchanged.
             * Existing audio behavior is 100% unchanged when tts != "none"
               (absent or any other value), so /test/index.html is unaffected.
             * No signature change — tts was already a form field on both routes.
v2.00.0013 - Fix streaming TTS timeout and error handling:
             * Replaced single 60s timeout with separate connect/read timeouts.
               httpx read timeout applies per-chunk for streaming, so long
               responses no longer time out mid-stream.
             * Added try/except inside streaming generators — errors now log
               and raise cleanly instead of silently dropping the stream.
             * Fixes PC/Chrome hang where TTS stream never completed.
             * OpenAI TTS now streams audio chunks as they arrive — playback
               starts in ~0.5-1s instead of waiting for the full file.
             * ElevenLabs TTS also streams via their streaming endpoint.
             * Coach text is sent as the FIRST chunk in a simple framing protocol:
               - Chunk 1: 4-byte big-endian length prefix + UTF-8 coach text
               - Remaining chunks: raw audio bytes
             * Frontend reads chunk 1 to get the text (displays immediately),
               then feeds the rest into the Web Audio API for streaming playback.
             * process_coaching() split into get_coach_text() + stream_tts() so
               text is available before TTS begins.
             * SILENCE_DURATION on VAD reduced recommendation: see frontend.
v2.00.0009 - Header version sync. Added POST /logs/clear.
v2.00.0004 - Dual STT support + CORS expose_headers fix.
v2.00.0003 - Removed primer fallback logic.
v2.00.0002 - Conversation history + primer-as-full-prompt + prompt caching.
"""

import os
import io
import struct
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
    if philosophy and philosophy_hash:
        if philosophy_hash not in philosophy_cache:
            philosophy_cache[philosophy_hash] = philosophy
            log("info", "CACHE", f"New philosophy cached — hash={philosophy_hash[:8]} len={len(philosophy)}")
        return philosophy_cache[philosophy_hash]
    if philosophy_hash and philosophy_hash in philosophy_cache:
        return philosophy_cache[philosophy_hash]
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Coach-Text"],
)


def make_safe_header(text: str) -> str:
    return text.replace("\r", " ").replace("\n", " ").encode("ascii", errors="replace").decode("ascii")


# --- STT (Whisper) ---
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


# --- Get coach text only (no TTS) ---
async def get_coach_text(user_text: str, ai: str, conversation_id: str,
                         philosophy_text: str, philosophy_hash: str,
                         keys: dict) -> str:
    """Run AI, update conversation history, return coach text."""
    conv = get_conversation(conversation_id, philosophy_hash)
    loop = asyncio.get_event_loop()

    conv["messages"].append({"role": "user", "content": user_text})

    if ai == "gemini":
        coach_text = await run_gemini(conv["messages"], philosophy_text, keys["gemini"])
    else:
        coach_text = await loop.run_in_executor(None, run_claude, conv["messages"], philosophy_text, keys["anthropic"])

    conv["messages"].append({"role": "assistant", "content": coach_text})
    trim_history(conv)
    return coach_text


# --- Streaming TTS generators ---

async def openai_tts_stream(text: str, api_key: str):
    """
    Yields the framing header first (4-byte length + UTF-8 coach text),
    then streams raw MP3 audio chunks from OpenAI TTS as they arrive.
    """
    text_bytes = text.encode("utf-8")
    yield struct.pack(">I", len(text_bytes)) + text_bytes
    log("info", "TTS:OpenAI", f"Streaming TTS for {len(text)} chars")

    try:
        # Use separate connect/read timeouts for streaming — read timeout applies
        # per-chunk, not for the entire stream, so long responses don't time out.
        timeout = httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                "https://api.openai.com/v1/audio/speech",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": "tts-1", "voice": "onyx", "input": text, "response_format": "mp3"},
            ) as response:
                response.raise_for_status()
                total = 0
                async for chunk in response.aiter_bytes(chunk_size=4096):
                    total += len(chunk)
                    yield chunk
                log("info", "TTS:OpenAI", f"Stream complete: {total} bytes")
    except Exception as e:
        log("error", "TTS:OpenAI", f"Stream error: {str(e)}", type(e).__name__)
        raise


async def elevenlabs_tts_stream(text: str, api_key: str):
    """
    Same framing protocol — header chunk first, then audio chunks.
    """
    voice_id = "pNInz6obpgDQGcFmaJgB"
    text_bytes = text.encode("utf-8")
    yield struct.pack(">I", len(text_bytes)) + text_bytes
    log("info", "TTS:ElevenLabs", f"Streaming TTS for {len(text)} chars")

    try:
        timeout = httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream",
                headers={"xi-api-key": api_key, "Content-Type": "application/json"},
                json={
                    "text": text,
                    "model_id": "eleven_turbo_v2_5",
                    "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}
                },
            ) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    log("error", "TTS:ElevenLabs", f"HTTP {response.status_code}", body[:300].decode("utf-8", errors="replace"))
                    response.raise_for_status()
                total = 0
                async for chunk in response.aiter_bytes(chunk_size=4096):
                    total += len(chunk)
                    yield chunk
                log("info", "TTS:ElevenLabs", f"Stream complete: {total} bytes")
    except Exception as e:
        log("error", "TTS:ElevenLabs", f"Stream error: {str(e)}", type(e).__name__)
        raise


def get_keys() -> dict:
    return {
        "anthropic":  os.getenv("ANTHROPIC_API_KEY"),
        "openai":     os.getenv("OPENAI_API_KEY"),
        "gemini":     os.getenv("GEMINI_API_KEY"),
        "elevenlabs": os.getenv("ELEVENLABS_API_KEY"),
    }


@app.get("/")
def read_root():
    return {"engine": "Coach Engine", "status": "operational", "version": "v2.00.0014"}


@app.get("/logs")
def get_logs(n: int = 50):
    return JSONResponse(content=log_buffer[-n:])


@app.post("/logs/clear")
def clear_logs():
    log_buffer.clear()
    log("info", "LOGS", "Log buffer cleared")
    return {"ok": True}


# --- WHISPER PATH: audio in, streaming audio out (or text-only when tts="none") ---
@app.post("/upload-audio")
async def handle_audio_coaching(
    file:             UploadFile = File(...),
    ai:               str = Form(default="claude"),
    tts:              str = Form(default="openai"),
    conversation_id:  str = Form(default="default"),
    philosophy:       str = Form(default=""),
    philosophy_hash:  str = Form(default="")
):
    log("info", "REQUEST", f"[audio] ai={ai} tts={tts} conv={conversation_id[:8]} hash={philosophy_hash[:8] if philosophy_hash else 'none'}")

    allowed_extensions = ["mp3", "wav", "m4a", "webm"]
    file_extension = file.filename.split(".")[-1].lower()
    if file_extension not in allowed_extensions:
        log("error", "REQUEST", f"Unsupported format: {file_extension}")
        raise HTTPException(status_code=400, detail="Unsupported audio format.")

    philosophy_text = resolve_philosophy(philosophy, philosophy_hash)
    if philosophy_text is None:
        log("error", "PRIMER", "No primer available — rejecting request")
        raise HTTPException(status_code=422, detail="No primer document available.")

    keys = get_keys()
    try:
        audio_bytes = await file.read()
        user_text   = await whisper_transcribe(audio_bytes, f"user_voice.{file_extension}", keys["openai"])
        coach_text  = await get_coach_text(user_text, ai, conversation_id, philosophy_text, philosophy_hash, keys)

        # --- TEXT-ONLY PATH (tts="none"): skip TTS, return JSON text ---
        # Used by typed/voice-typed modes that want a fast text reply with no
        # audio synthesis. We still ran STT above (voice came in as audio) so the
        # transcript is included, then the coach text is returned as JSON.
        if tts == "none":
            log("info", "COMPLETE", f"Text-only reply (tts=none) — {len(coach_text)} chars")
            return JSONResponse(content={
                "coach_text": coach_text,
                "conversation_id": conversation_id,
                "transcript": user_text,
            })

        # --- STREAMING AUDIO PATH (unchanged) ---
        if tts == "elevenlabs":
            stream_gen = elevenlabs_tts_stream(coach_text, keys["elevenlabs"])
            media_type = "audio/mpeg"
        else:
            stream_gen = openai_tts_stream(coach_text, keys["openai"])
            media_type = "audio/mpeg"  # OpenAI streaming returns mp3

        return StreamingResponse(stream_gen, media_type=media_type)

    except HTTPException:
        raise
    except Exception as e:
        log("error", "ERROR", str(e), type(e).__name__)
        raise HTTPException(status_code=500, detail=f"Engine fault: {str(e)}")


# --- NATIVE PATH: text in, streaming audio out (or text-only when tts="none") ---
@app.post("/coach-text")
async def handle_text_coaching(
    text:             str = Form(...),
    ai:               str = Form(default="claude"),
    tts:              str = Form(default="openai"),
    conversation_id:  str = Form(default="default"),
    philosophy:       str = Form(default=""),
    philosophy_hash:  str = Form(default="")
):
    log("info", "REQUEST", f"[text] ai={ai} tts={tts} conv={conversation_id[:8]} hash={philosophy_hash[:8] if philosophy_hash else 'none'}")
    log("info", "STT", f"Native transcript: {text}")

    philosophy_text = resolve_philosophy(philosophy, philosophy_hash)
    if philosophy_text is None:
        log("error", "PRIMER", "No primer available — rejecting request")
        raise HTTPException(status_code=422, detail="No primer document available.")

    keys = get_keys()
    try:
        coach_text = await get_coach_text(text.strip(), ai, conversation_id, philosophy_text, philosophy_hash, keys)

        # --- TEXT-ONLY PATH (tts="none"): skip TTS, return JSON text ---
        # The fast path for typed/voice-typed chat: no TTS call, no audio latency,
        # no TTS cost. Returns the coach text as JSON for the frontend to display.
        if tts == "none":
            log("info", "COMPLETE", f"Text-only reply (tts=none) — {len(coach_text)} chars")
            return JSONResponse(content={
                "coach_text": coach_text,
                "conversation_id": conversation_id,
            })

        # --- STREAMING AUDIO PATH (unchanged) ---
        if tts == "elevenlabs":
            stream_gen = elevenlabs_tts_stream(coach_text, keys["elevenlabs"])
            media_type = "audio/mpeg"
        else:
            stream_gen = openai_tts_stream(coach_text, keys["openai"])
            media_type = "audio/mpeg"

        return StreamingResponse(stream_gen, media_type=media_type)

    except HTTPException:
        raise
    except Exception as e:
        log("error", "ERROR", str(e), type(e).__name__)
        raise HTTPException(status_code=500, detail=f"Engine fault: {str(e)}")