"""
Discovery Coaching Engine
Location: /var/www/coach-engine/dev/main.py
Version: v1.7.0
Changes: Added AI engine toggle (Claude/Gemini) and TTS toggle (OpenAI/ElevenLabs).
         Frontend sends ai= and tts= form params to select provider per request.
"""

import os
import io
import asyncio
import httpx
import anthropic
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Coach Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_PHILOSOPHY = "financial_philosophy.md"


def get_coaching_philosophy() -> str:
    if os.path.exists(DEFAULT_PHILOSOPHY):
        with open(DEFAULT_PHILOSOPHY, "r", encoding="utf-8") as f:
            return f.read()
    return "You are a direct, no-fluff business and financial coach."


def make_safe_header(text: str) -> str:
    """Strip newlines and non-ASCII for HTTP header safety."""
    return text.replace("\r", " ").replace("\n", " ").encode("ascii", errors="replace").decode("ascii")


# --- STT ---
async def whisper_transcribe(audio_bytes: bytes, filename: str, api_key: str) -> str:
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": (filename, audio_bytes)},
            data={"model": "whisper-1"},
        )
        response.raise_for_status()
        return response.json()["text"].strip()


# --- AI: Claude ---
def run_claude(user_text: str, philosophy: str, api_key: str) -> str:
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
    return message.content[0].text.strip()


# --- AI: Gemini ---
async def run_gemini(user_text: str, philosophy: str, api_key: str) -> str:
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}",
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
        response.raise_for_status()
        return response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()


# --- TTS: OpenAI ---
async def openai_tts(text: str, api_key: str) -> bytes:
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
        return response.content


# --- TTS: ElevenLabs ---
async def elevenlabs_tts(text: str, api_key: str) -> bytes:
    # Using Adam voice — deep, authoritative male voice
    voice_id = "pNInz6obpgDQGcFmaJgB"
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
            headers={
                "xi-api-key": api_key,
                "Content-Type": "application/json",
            },
            json={
                "text": text,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75
                }
            },
        )
        response.raise_for_status()
        return response.content


@app.get("/")
def read_root():
    return {"engine": "Coach Engine", "status": "operational"}


@app.post("/upload-audio")
async def handle_audio_coaching(
    file: UploadFile = File(...),
    ai:  str = Form(default="claude"),
    tts: str = Form(default="openai")
):
    allowed_extensions = ["mp3", "wav", "m4a", "webm"]
    file_extension = file.filename.split(".")[-1].lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Unsupported audio format.")

    anthropic_key  = os.getenv("ANTHROPIC_API_KEY")
    openai_key     = os.getenv("OPENAI_API_KEY")
    gemini_key     = os.getenv("GEMINI_API_KEY")
    elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")

    try:
        audio_bytes = await file.read()
        philosophy  = get_coaching_philosophy()
        loop        = asyncio.get_event_loop()

        # Step 1: Whisper STT (always)
        filename  = f"user_voice.{file_extension}"
        user_text = await whisper_transcribe(audio_bytes, filename, openai_key)
        print(f"[STT] Transcript: {user_text}")

        # Step 2: AI response (Claude or Gemini)
        if ai == "gemini":
            coach_text = await run_gemini(user_text, philosophy, gemini_key)
        else:
            coach_text = await loop.run_in_executor(None, run_claude, user_text, philosophy, anthropic_key)
        print(f"[AI:{ai}] Response: {coach_text}")

        # Step 3: TTS (OpenAI or ElevenLabs)
        if tts == "elevenlabs":
            audio_data = await elevenlabs_tts(coach_text, elevenlabs_key)
            media_type = "audio/mpeg"
        else:
            audio_data = await openai_tts(coach_text, openai_key)
            media_type = "audio/wav"
        print(f"[TTS:{tts}] Complete: {len(audio_data)} bytes")

        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type=media_type,
            headers={"X-Coach-Text": make_safe_header(coach_text)}
        )

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        raise HTTPException(status_code=500, detail=f"Engine fault: {str(e)}")