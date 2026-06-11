"""
Discovery Coaching Engine
Location: /home/discovery/fastapi_test/main.py
Version: v1.6.0
Changes: Fixed illegal HTTP header value — strip newlines and encode
         coach text as ASCII-safe before setting X-Coach-Text header.
"""

import os
import io
import asyncio
import httpx
import anthropic
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Discovery Coaching Engine")

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
    """Strip newlines and non-ASCII so h11 doesn't reject the header."""
    return text.replace("\r", " ").replace("\n", " ").encode("ascii", errors="replace").decode("ascii")


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


@app.get("/")
def read_root():
    return {"engine": "Discovery Coaching System", "status": "operational"}


@app.post("/upload-audio")
async def handle_audio_coaching(file: UploadFile = File(...)):
    allowed_extensions = ["mp3", "wav", "m4a", "webm"]
    file_extension = file.filename.split(".")[-1].lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Unsupported audio format.")

    claude_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if not claude_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not set.")
    if not openai_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set.")

    try:
        audio_bytes = await file.read()
        philosophy  = get_coaching_philosophy()

        # Step 1: Whisper STT
        filename  = f"user_voice.{file_extension}"
        user_text = await whisper_transcribe(audio_bytes, filename, openai_key)
        print(f"Transcript: {user_text}")

        # Step 2: Claude coached response
        loop       = asyncio.get_event_loop()
        coach_text = await loop.run_in_executor(None, run_claude, user_text, philosophy, claude_key)
        print(f"Coach response: {coach_text}")

        # Step 3: OpenAI TTS
        audio_data = await openai_tts(coach_text, openai_key)
        print(f"TTS complete: {len(audio_data)} bytes")

        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type="audio/wav",
            headers={"X-Coach-Text": make_safe_header(coach_text)}
        )

    except Exception as e:
        print(f"API Execution exception caught: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Engine processing fault: {str(e)}")
