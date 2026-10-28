from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Form,
    Depends,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional
from uuid import uuid4
import io
import asyncio
import base64
import time

from app.core.agent import AIAgent
from app.core.tts import TextToSpeech
from app.core.stt import SpeechToText
from app.services.llm_service import get_llm_service
from app.config import WEBSOCKET_SYNTHESIZE_AUDIO

router = APIRouter(prefix="/voice", tags=["voice"])

class SourceInfo(BaseModel):
    """Source information model"""
    title: str
    url: Optional[str] = None
    snippet: Optional[str] = None

class VoiceResponse(BaseModel):
    """Voice response model (text part)"""
    message_id: str
    text: str
    sources: List[SourceInfo] = Field(default_factory=list)
    session_id: str

@router.post("/", response_model=VoiceResponse)
async def process_voice(
    audio: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    llm_service = Depends(get_llm_service)
):
    """
    Process voice input and return text response

    - **audio**: Audio file with speech input
    - **session_id**: Session identifier for conversation context (optional)
    """
    # Generate a session ID if not provided
    session_id = session_id or str(uuid4())

    try:
        # Read audio file
        audio_data = await audio.read()

        # Convert speech to text
        stt = SpeechToText()
        text_input = stt.transcribe(audio_data)

        if not text_input:
            raise HTTPException(status_code=400, detail="Could not transcribe audio")

        # Process text with AI agent
        agent = AIAgent(llm_service)
        response, sources = await agent.process_message(
            message=text_input,
            session_id=session_id,
            interaction_type="voice",
        )

        return VoiceResponse(
            message_id=str(uuid4()),
            text=response,
            sources=sources,
            session_id=session_id
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing voice: {str(e)}")

@router.post("/tts")
async def text_to_speech(text: str = Form(...), voice: Optional[str] = Form("default")):
    """
    Convert text to speech

    - **text**: Text to convert to speech
    - **voice**: Voice identifier (optional)
    """
    try:
        tts = TextToSpeech()
        audio_data = tts.synthesize(text, voice)

        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type=(
                tts.media_type
                if isinstance(getattr(tts, "media_type", None), str)
                else "audio/mpeg"
            )
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error converting text to speech: {str(e)}")

@router.post("/stt")
async def speech_to_text(audio: UploadFile = File(...)):
    """
    Convert speech to text

    - **audio**: Audio file with speech input
    """
    try:
        # Read audio file
        audio_data = await audio.read()

        # Convert speech to text
        stt = SpeechToText()
        text = stt.transcribe(audio_data)

        if not text:
            raise HTTPException(status_code=400, detail="Could not transcribe audio")

        return {"text": text}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error converting speech to text: {str(e)}")

@router.websocket("/ws/{session_id}")
async def realtime_voice(websocket: WebSocket, session_id: str):
    """Handle realtime text or complete base64-encoded audio turns.

    The event contract is documented in `docs/ARCHITECTURE.md`. Provider calls
    run in worker threads so speech SDKs do not block FastAPI's event loop.
    """
    await websocket.accept()
    try:
        agent = AIAgent(get_llm_service())
        stt = SpeechToText()
        tts = TextToSpeech() if WEBSOCKET_SYNTHESIZE_AUDIO else None
        while True:
            event = await websocket.receive_json()
            event_type = event.get("type")
            started = time.perf_counter()

            if event_type == "text":
                text = str(event.get("text", "")).strip()
            elif event_type == "audio":
                try:
                    audio = base64.b64decode(event.get("audio", ""), validate=True)
                except (ValueError, TypeError):
                    await websocket.send_json({
                        "type": "error",
                        "error": "audio must be valid base64",
                    })
                    continue
                text = await asyncio.to_thread(stt.transcribe, audio)
                await websocket.send_json({
                    "type": "transcript",
                    "text": text,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                })
            elif event_type == "ping":
                await websocket.send_json({"type": "pong"})
                continue
            else:
                await websocket.send_json({
                    "type": "error",
                    "error": "supported event types: text, audio, ping",
                })
                continue

            if not text:
                await websocket.send_json({"type": "error", "error": "empty input"})
                continue

            response, sources = await agent.process_message(
                text,
                session_id,
                interaction_type="voice",
            )
            payload = {
                "type": "response",
                "text": response,
                "sources": sources,
                "session_id": session_id,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            }
            await websocket.send_json(payload)

            if tts:
                audio = await asyncio.to_thread(tts.synthesize, response, "default")
                await websocket.send_json({
                    "type": "audio",
                    "audio": base64.b64encode(audio).decode("ascii"),
                    "media_type": tts.media_type,
                })
    except WebSocketDisconnect:
        return
    except Exception as exc:
        await websocket.send_json({"type": "error", "error": str(exc)})
        await websocket.close(code=1011)
