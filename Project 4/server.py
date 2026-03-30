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
from urllib.parse import urlparse

import aiohttp

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

    async def deserialize(self, data: str | bytes):
        try:
            message = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            return await super().deserialize(data)

        if message.get("event") == "start" and not self._stream_initialized:
            start = message.get("start", {})
            self._stream_id = start.get("streamId", "")
            self._call_id = start.get("callId")
            self._stream_initialized = True
            logger.info(
                f"[Plivo] stream started — stream_id={self._stream_id}  call_id={self._call_id}"
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


# ── Farewell detector ─────────────────────────────────────────────────────────

_FAREWELL_KEYWORDS = {"bye", "goodbye", "good bye", "thank you", "thanks", "that's all", "that is all"}

class FarewellDetector(FrameProcessor):
    """Intercepts farewell transcriptions, speaks a closing message, then ends the call.

    Placement: between stt and context_aggregator.user() so that farewell
    transcriptions are absorbed before the LLM sees them.  The closing TextFrame
    is injected via task.queue_frames() which routes directly to TTS, bypassing
    the LLM.  The pipeline is cancelled after a short delay to let TTS finish.
    """

    def __init__(self):
        super().__init__()
        self._task_ref = None
        self._farewell_sent = False

    def set_task(self, task):
        self._task_ref = task

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame) and not self._farewell_sent:
            text_lower = frame.text.lower()
            if any(kw in text_lower for kw in _FAREWELL_KEYWORDS):
                self._farewell_sent = True
                logger.info(f"[Farewell] detected in: '{frame.text}'")
                await self._task_ref.queue_frames([
                    TextFrame("Thank you for calling, have a great day!")
                ])
                # Wait for ElevenLabs to stream audio back through Plivo before
                # shutting down.  EndFrame drains the pipeline gracefully (unlike
                # task.cancel() which tears it down immediately).
                asyncio.create_task(self._end_after_delay(6.0))
                return  # absorb — don't forward to LLM

        await self.push_frame(frame, direction)

    async def _end_after_delay(self, delay: float):
        await asyncio.sleep(delay)
        if self._task_ref:
            await self._task_ref.queue_frames([EndFrame()])


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

    async with aiohttp.ClientSession() as aiohttp_session:
        transport = FastAPIWebsocketTransport(
            websocket=websocket,
            params=FastAPIWebsocketParams(
                serializer=DynamicPlivoSerializer(),
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
                voice=os.getenv("ELEVENLABS_VOICE_ID", "TX3LPaxmHKxFdv7VOQHJ"),
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

        farewell_detector = FarewellDetector()
        stt_error_handler = STTErrorHandler()

        # Pipeline: audio in → speech-to-text → LLM → text-to-speech → audio out
        pipeline = Pipeline(
            [
                transport.input(),          # receives μ-law audio from Plivo, runs VAD
                stt,                        # Deepgram: audio → transcript
                stt_error_handler,          # speaks fallback if STT fails, absorbs ErrorFrame
                farewell_detector,          # intercepts goodbye/thanks, speaks farewell, ends call
                context_aggregator.user(),  # accumulates user turn, fires when turn ends
                llm,                        # GPT-4.1-mini: text → text response
                tts,                        # ElevenLabs: text → audio frames
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

        farewell_detector.set_task(task)
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
