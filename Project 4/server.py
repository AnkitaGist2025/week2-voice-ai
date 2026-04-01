"""
Day 4 – Pipecat + Plivo phone calls  (single-port edition)
===========================================================
Both HTTP and WebSocket run on the same port (5000) via FastAPI + uvicorn,
so only ONE ngrok tunnel is needed.

Endpoints
---------
  GET/POST /answer   – Plivo calls this on answer; returns XML that tells
                       Plivo to open a WebSocket to /ws on the same host.
  WS       /ws       – Plivo streams bidirectional μ-law audio here.

Call flow
---------
  Plivo dials in → GET /answer → XML: <Stream>wss://NGROK_URL/ws</Stream>
  Plivo opens WebSocket to /ws
  DynamicPlivoSerializer reads "start" event → captures stream_id / call_id
  Plivo streams μ-law audio → SileroVAD → Deepgram STT → GPT-4.1-mini → ElevenLabs TTS
  Bot audio is μ-law-encoded and sent back over the same WebSocket

Run
---
  uvicorn server:app --host 0.0.0.0 --port 5000
"""

import asyncio
import audioop
import base64
import json
import os
from datetime import datetime, timezone
from urllib.parse import urlparse

import aiohttp
import asyncpg

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.responses import Response
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    EndFrame,
    ErrorFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.serializers.plivo import PlivoFrameSerializer
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService, ElevenLabsTTSSettings
from pipecat.transports.websocket.fastapi import FastAPIWebsocketTransport, FastAPIWebsocketParams

load_dotenv()

app = FastAPI()


# ── HTTP endpoints ────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.api_route("/answer", methods=["GET", "POST"])
async def answer():
    """
    Plivo Answer URL handler.
    Derives the WebSocket URL from SERVER_URL by swapping the scheme and
    appending /ws — so both HTTP and WS share the same ngrok tunnel.
    """
    server_url = os.getenv("SERVER_URL", "").rstrip("/")
    host = urlparse(server_url).netloc or server_url
    ws_url = f"wss://{host}/ws"

    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Stream streamTimeout="86400"
            contentType="audio/x-mulaw;rate=8000"
            bidirectional="true"
            keepCallAlive="true">
        {ws_url}
    </Stream>
</Response>"""

    logger.info(f"/answer → streaming to {ws_url}")
    return Response(content=xml, media_type="text/xml")


# ── Dynamic Plivo serializer ──────────────────────────────────────────────────

class DynamicPlivoSerializer(PlivoFrameSerializer):
    """PlivoFrameSerializer that self-configures from the Plivo 'start' event.

    Plivo's first WebSocket message is always a 'start' event that carries the
    streamId and callId we need.  The base class requires these at construction
    time, so we intercept that first message here and patch them in before
    routing any subsequent frames to the base class logic.
    """

    def __init__(self):
        super().__init__(
            stream_id="",                            # filled in on first 'start' event
            auth_id=os.getenv("PLIVO_AUTH_ID"),
            auth_token=os.getenv("PLIVO_AUTH_TOKEN"),
        )
        self._stream_initialized = False
        self._ratecv_state = None  # audioop.ratecv conversion state
        self.caller_number = "unknown"

    async def deserialize(self, data: str | bytes):
        try:
            message = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            return await super().deserialize(data)

        if message.get("event") == "start" and not self._stream_initialized:
            start = message.get("start", {})
            self._stream_id = start.get("streamId", "")
            self._call_id = start.get("callId")
            self.caller_number = start.get("from", "unknown")
            self._stream_initialized = True
            logger.info(f"[Plivo] start event payload: {json.dumps(start)}")
            logger.info(
                f"[Plivo] stream started — stream_id={self._stream_id}  "
                f"call_id={self._call_id}  from={self.caller_number}"
            )
            return None  # 'start' carries no audio

        if message.get("event") == "stop":
            logger.info("[Plivo] stream stopped")
            return None

        return await super().deserialize(data)

    async def serialize(self, frame):
        from pipecat.frames.frames import AudioRawFrame
        if not isinstance(frame, AudioRawFrame):
            return await super().serialize(frame)

        # Use audioop.ratecv which always produces output and maintains state
        # between calls — unlike the streaming resampler which can return empty
        # bytes when the chunk doesn't align with its internal buffer.
        resampled, self._ratecv_state = audioop.ratecv(
            frame.audio, 2, 1, frame.sample_rate, 8000, self._ratecv_state
        )
        if not resampled:
            return None

        ulaw_bytes = audioop.lin2ulaw(resampled, 2)
        payload = base64.b64encode(ulaw_bytes).decode("utf-8")
        message = {
            "event": "playAudio",
            "media": {
                "contentType": "audio/x-mulaw",
                "sampleRate": 8000,
                "payload": payload,
            },
            "streamId": self._stream_id,
        }
        return json.dumps(message)


# ── Audio debug logger ────────────────────────────────────────────────────────

class AudioDebugLogger(FrameProcessor):
    """Logs key pipeline events to help debug audio flow.

    Placement: between tts and transport.output() so it observes both
    TTS-produced audio frames (flowing from tts) and upstream frames
    (STT transcriptions, LLM events) that travel the full pipeline length.
    """

    def __init__(self):
        super().__init__()
        self._tts_chunks = 0
        self._tts_bytes = 0

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            logger.info(f"[STT ] transcript: '{frame.text}'")

        elif isinstance(frame, LLMFullResponseStartFrame):
            logger.info("[LLM ] response started")

        elif isinstance(frame, LLMFullResponseEndFrame):
            logger.info("[LLM ] response ended")

        elif isinstance(frame, TTSStartedFrame):
            self._tts_chunks = 0
            self._tts_bytes = 0
            logger.info("[TTS ] synthesis started")

        elif isinstance(frame, TTSAudioRawFrame):
            self._tts_chunks += 1
            self._tts_bytes += len(frame.audio)
            if self._tts_chunks == 1:
                logger.info(
                    f"[TTS ] first audio chunk — {len(frame.audio)} bytes "
                    f"@ {frame.sample_rate}Hz"
                )

        elif isinstance(frame, TTSStoppedFrame):
            logger.info(
                f"[TTS ] synthesis done — {self._tts_chunks} chunks, "
                f"{self._tts_bytes} bytes total sent to transport"
            )

        await self.push_frame(frame, direction)


# ── Farewell handling ─────────────────────────────────────────────────────────

_FAREWELL_KEYWORDS = {"bye", "goodbye", "good bye", "thank you", "thanks", "that's all", "that is all"}


class _FarewellState:
    """Shared flag between FarewellDetector (input side) and FarewellShutdown (output side)."""
    def __init__(self):
        self.farewell_sent = False
        self.tts_started = False  # True once TTSStartedFrame fires after farewell


class FarewellDetector(FrameProcessor):
    """Intercepts farewell transcriptions and replaces them with an LLM instruction.

    Placement: between stt and context_aggregator.user().  When the caller says
    goodbye the original TranscriptionFrame is replaced with a directive so the
    normal LLM → TTS path produces the farewell audio.  No task reference needed
    here — shutdown is handled downstream by FarewellShutdown.
    """

    def __init__(self, state: _FarewellState):
        super().__init__()
        self._state = state

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame) and not self._state.farewell_sent:
            text_lower = frame.text.lower()
            if any(kw in text_lower for kw in _FAREWELL_KEYWORDS):
                self._state.farewell_sent = True
                logger.info(f"[Farewell] detected: '{frame.text}' — injecting LLM instruction")
                # Replace the caller's words with an instruction; let LLM → TTS handle it.
                frame = TranscriptionFrame(
                    text="The caller said goodbye. Respond with a warm farewell and nothing else.",
                    user_id=frame.user_id,
                    timestamp=frame.timestamp,
                )

        await self.push_frame(frame, direction)


class FarewellShutdown(FrameProcessor):
    """Ends the pipeline once the farewell TTS synthesis is fully complete.

    Placement: after tts, before transport.output().  Watches for TTSStartedFrame
    (to confirm the farewell TTS has begun) and TTSStoppedFrame (to confirm all
    audio chunks have been pushed to the transport) before queuing EndFrame.
    """

    def __init__(self, state: _FarewellState):
        super().__init__()
        self._state = state
        self._task_ref = None

    def set_task(self, task):
        self._task_ref = task

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if self._state.farewell_sent:
            if isinstance(frame, TTSStartedFrame):
                self._state.tts_started = True

            elif isinstance(frame, TTSStoppedFrame) and self._state.tts_started:
                # Push the stop frame so transport.output() can cleanly finish,
                # then signal graceful pipeline shutdown.
                await self.push_frame(frame, direction)
                logger.info("[Farewell] TTS done — queuing EndFrame to close call")
                if self._task_ref:
                    await self._task_ref.queue_frames([EndFrame()])
                return  # already pushed above

        await self.push_frame(frame, direction)


# ── STT error handler ─────────────────────────────────────────────────────────

class STTErrorHandler(FrameProcessor):
    """Catches ErrorFrames from Deepgram STT and speaks a polite fallback.

    Placement: immediately after stt so STT errors are handled before
    they propagate and potentially crash downstream processors.
    """

    def __init__(self):
        super().__init__()
        self._task_ref = None

    def set_task(self, task):
        self._task_ref = task

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, ErrorFrame):
            logger.warning(f"[STT ] error — {frame.error}")
            if self._task_ref:
                await self._task_ref.queue_frames([
                    TextFrame("I'm having trouble hearing you, could you repeat that?")
                ])
            return  # absorb the ErrorFrame so it doesn't crash the pipeline

        await self.push_frame(frame, direction)


# ── Transcript collector ──────────────────────────────────────────────────────

class TranscriptCollector(FrameProcessor):
    """Accumulates all STT transcription text for post-call logging.

    Placement: after stt_error_handler, before farewell_detector, so it
    records what the caller actually said (not the injected LLM instruction).
    """

    def __init__(self):
        super().__init__()
        self.lines: list[str] = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            self.lines.append(frame.text.strip())
        await self.push_frame(frame, direction)


# ── Call logging ──────────────────────────────────────────────────────────────

async def _log_call_to_db(
    caller_number: str,
    duration_secs: int,
    transcript: str,
    timestamp: datetime,
) -> None:
    """Insert a call record into PostgreSQL.  Called as a background task so it
    never blocks the WebSocket handler or affects the next incoming call."""
    raw_url = os.getenv("DATABASE_URL", "")
    if not raw_url:
        logger.warning("[DB] DATABASE_URL not set — skipping call log")
        return

    # Railway exposes postgres:// but asyncpg requires postgresql://
    db_url = raw_url.replace("postgres://", "postgresql://", 1)

    try:
        conn = await asyncpg.connect(db_url)
        try:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS call_logs (
                    id                   SERIAL PRIMARY KEY,
                    caller_number        TEXT,
                    call_duration_seconds INTEGER,
                    full_transcript      TEXT,
                    timestamp            TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            await conn.execute(
                "INSERT INTO call_logs "
                "(caller_number, call_duration_seconds, full_transcript, timestamp) "
                "VALUES ($1, $2, $3, $4)",
                caller_number, duration_secs, transcript, timestamp,
            )
            logger.info(
                f"[DB] call logged — caller={caller_number}  "
                f"duration={duration_secs}s  transcript_chars={len(transcript)}"
            )
        finally:
            await conn.close()
    except Exception as exc:
        logger.error(f"[DB] failed to log call: {exc}")


# ── WebSocket endpoint + Pipecat pipeline ────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Plivo connects here after receiving the /answer XML.
    A fresh pipeline is created for every call, so concurrent calls are
    naturally isolated.
    """
    await websocket.accept()
    logger.info(f"[Plivo] WebSocket connected from {websocket.client}")
    call_start = datetime.now(timezone.utc)

    async with aiohttp.ClientSession() as aiohttp_session:
        serializer = DynamicPlivoSerializer()
        transport = FastAPIWebsocketTransport(
            websocket=websocket,
            params=FastAPIWebsocketParams(
                serializer=serializer,
                audio_in_enabled=True,
                audio_out_enabled=True,
                add_wav_header=False,
                # SileroVAD runs on-device to detect end-of-speech before sending to STT
                vad_analyzer=SileroVADAnalyzer(
                    params=VADParams(
                        # Raised confidence + min_volume to ignore Plivo codec noise
                        # at stream start, which was triggering false interruptions.
                        confidence=0.85,
                        start_secs=0.4,   # must hear speech for 400ms before triggering
                        stop_secs=0.8,
                        min_volume=0.6,
                    )
                ),
            ),
        )

        stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4.1-mini",
        )

        tts = ElevenLabsTTSService(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            settings=ElevenLabsTTSSettings(
                voice_id=os.getenv("ELEVENLABS_VOICE_ID", "TX3LPaxmHKxFdv7VOQHJ"),
                model="eleven_turbo_v2_5",
            ),
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a friendly phone assistant. "
                    "Keep every reply to 1 sentence max — your words are spoken aloud. "
                    "Be direct and conversational. "
                    "Never use markdown, bullet points, numbers, or special characters."
                ),
            }
        ]

        context = OpenAILLMContext(messages=messages)
        context_aggregator = LLMContextAggregatorPair(context)

        farewell_state = _FarewellState()
        farewell_detector = FarewellDetector(farewell_state)
        farewell_shutdown = FarewellShutdown(farewell_state)
        stt_error_handler = STTErrorHandler()
        transcript_collector = TranscriptCollector()

        # Pipeline: audio in → speech-to-text → LLM → text-to-speech → audio out
        pipeline = Pipeline(
            [
                transport.input(),          # receives μ-law audio from Plivo, runs VAD
                stt,                        # Deepgram: audio → transcript
                stt_error_handler,          # speaks fallback if STT fails, absorbs ErrorFrame
                transcript_collector,       # records what the caller said for post-call logging
                farewell_detector,          # replaces goodbye transcript with LLM instruction
                context_aggregator.user(),  # accumulates user turn, fires when turn ends
                llm,                        # GPT-4.1-mini: text → text response
                tts,                        # ElevenLabs: text → audio frames
                farewell_shutdown,          # ends call after farewell TTS synthesis completes
                AudioDebugLogger(),         # logs STT/LLM/TTS events for debugging
                transport.output(),         # μ-law-encodes audio, sends back to Plivo
                context_aggregator.assistant(),
            ]
        )

        task = PipelineTask(
            pipeline,
            params=PipelineParams(allow_interruptions=True),
            idle_timeout_secs=300,
        )

        farewell_shutdown.set_task(task)
        stt_error_handler.set_task(task)

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, websocket):
            logger.info("[Plivo] WebSocket connected — queuing greeting")
            await task.queue_frames([
                TextFrame("Hello, thank you for calling. How can I help you today?")
            ])

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, websocket):
            logger.info("[Plivo] WebSocket disconnected — cancelling pipeline")
            await task.cancel()

        runner = PipelineRunner()
        await runner.run(task)

        # ── Post-call logging (non-blocking) ──────────────────────────────────
        call_end = datetime.now(timezone.utc)
        duration_secs = int((call_end - call_start).total_seconds())
        full_transcript = " ".join(transcript_collector.lines)
        asyncio.create_task(
            _log_call_to_db(
                caller_number=serializer.caller_number,
                duration_secs=duration_secs,
                transcript=full_transcript,
                timestamp=call_end,
            )
        )
