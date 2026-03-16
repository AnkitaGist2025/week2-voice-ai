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

import json
import os

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.responses import Response
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.serializers.plivo import PlivoFrameSerializer
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
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
    ws_url = server_url.replace("https://", "wss://").replace("http://", "ws://") + "/ws"

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
                    confidence=0.7,
                    start_secs=0.2,
                    stop_secs=0.8,
                    min_volume=0.4,
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
        voice_id=os.getenv("ELEVENLABS_VOICE_ID", "TX3LPaxmHKxFdv7VOQHJ"),
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a friendly phone assistant. "
                "Keep every reply to 1–2 sentences — your words are spoken aloud. "
                "Avoid markdown, bullet points, or any special characters."
            ),
        }
    ]

    context = OpenAILLMContext(messages=messages)
    context_aggregator = LLMContextAggregatorPair(context)

    # Pipeline: audio in → speech-to-text → LLM → text-to-speech → audio out
    pipeline = Pipeline(
        [
            transport.input(),          # receives μ-law audio from Plivo, runs VAD
            stt,                        # Deepgram: audio → transcript
            context_aggregator.user(),  # accumulates user turn, fires when turn ends
            llm,                        # GPT-4.1-mini: text → text response
            tts,                        # ElevenLabs: text → audio frames
            transport.output(),         # μ-law-encodes audio, sends back to Plivo
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(allow_interruptions=True),
        idle_timeout_secs=300,
    )

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, websocket):
        logger.info("[Plivo] WebSocket disconnected — cancelling pipeline")
        await task.cancel()

    runner = PipelineRunner()
    await runner.run(task)
