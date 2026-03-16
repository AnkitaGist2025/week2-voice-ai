import argparse
import asyncio
import datetime
import os
import random
import time

import pyaudio
from dotenv import load_dotenv
from pipecat.audio.interruptions.min_words_interruption_strategy import MinWordsInterruptionStrategy
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import TTSAudioRawFrame, UserStoppedSpeakingFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair, LLMUserAggregatorParams
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

load_dotenv()


class LatencyTracker(FrameProcessor):
    """Measures end-to-end latency: VAD end-of-speech → first TTS audio chunk.

    Place this processor between the TTS service and transport.output() so it
    sees both the UserStoppedSpeakingFrame (flowing downstream from the
    transport input) and the TTSAudioRawFrame (produced by the TTS service).
    """

    def __init__(self, model_name: str):
        super().__init__()
        self._model_name = model_name
        self._vad_end_time: float | None = None
        self._waiting_for_tts: bool = False
        self._exchange_count: int = 0
        self._latencies: list[float] = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStoppedSpeakingFrame):
            self._vad_end_time = time.perf_counter()
            self._waiting_for_tts = True
            ts = time.strftime("%H:%M:%S")
            print(f"\n[{ts}] VAD   end-of-speech detected")

        elif (
            isinstance(frame, TTSAudioRawFrame)
            and self._waiting_for_tts
            and self._vad_end_time is not None
        ):
            self._waiting_for_tts = False
            latency = time.perf_counter() - self._vad_end_time
            self._exchange_count += 1
            self._latencies.append(latency)
            avg = sum(self._latencies) / len(self._latencies)
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts}] TTS   first audio chunk ready")
            print(
                f"[Latency #{self._exchange_count}] "
                f"model={self._model_name}  "
                f"this={latency:.3f}s  "
                f"avg={avg:.3f}s  "
                f"(n={len(self._latencies)})"
            )

        await self.push_frame(frame, direction)

    def print_summary(self) -> None:
        if not self._latencies:
            print("\nNo exchanges recorded.")
            return
        avg = sum(self._latencies) / len(self._latencies)
        print(
            f"\n{'─'*50}\n"
            f"Latency summary  model={self._model_name}\n"
            f"  exchanges : {len(self._latencies)}\n"
            f"  min       : {min(self._latencies):.3f}s\n"
            f"  max       : {max(self._latencies):.3f}s\n"
            f"  avg       : {avg:.3f}s\n"
            f"{'─'*50}"
        )


# ── Tool schemas (sent to OpenAI so the LLM knows when to call each tool) ──────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Returns the current local time. Call this when the user asks what time it is.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tell_joke",
            "description": "Returns a random joke. Call this when the user asks for a joke.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "Looks up the status of an order by its order number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_number": {
                        "type": "string",
                        "description": "The order number to look up, e.g. '12345'",
                    }
                },
                "required": ["order_number"],
            },
        },
    },
]

# ── Tool implementation callbacks ───────────────────────────────────────────────
# Pipecat signature: (function_name, tool_call_id, arguments, llm, context, result_callback)
# Call result_callback(value) to send the result back into the LLM context.

_JOKES = [
    "Why don't scientists trust atoms? Because they make up everything.",
    "I told my wife she was drawing her eyebrows too high. She looked surprised.",
    "Why do cows wear bells? Because their horns don't work.",
    "I asked the librarian if they had books about paranoia. She whispered: 'They're right behind you.'",
    "Why did the scarecrow win an award? Because he was outstanding in his field.",
]

_ORDER_STATUSES = ["processing", "shipped", "out for delivery", "delivered", "delayed"]


async def handle_get_current_time(function_name, tool_call_id, arguments, llm, context, result_callback):
    current_time = datetime.datetime.now().strftime("%I:%M %p")
    print(f"[Tool] get_current_time → {current_time}")
    await result_callback({"time": current_time})


async def handle_tell_joke(function_name, tool_call_id, arguments, llm, context, result_callback):
    joke = random.choice(_JOKES)
    print(f"[Tool] tell_joke → {joke}")
    await result_callback({"joke": joke})


async def handle_lookup_order(function_name, tool_call_id, arguments, llm, context, result_callback):
    order_number = arguments.get("order_number", "unknown")
    status = random.choice(_ORDER_STATUSES)
    eta = random.randint(1, 5)
    result = {"order_number": order_number, "status": status, "estimated_days": eta}
    print(f"[Tool] lookup_order({order_number}) → {result}")
    await result_callback(result)


async def main(model: str) -> None:
    # Look up the system default output device so playback follows whatever
    # is selected in macOS System Settings (speakers, headphones, etc.).
    _pa = pyaudio.PyAudio()
    _default_out = _pa.get_default_output_device_info()
    output_device_index = int(_default_out["index"])
    print(f"Audio output device: {_default_out['name']} (index {output_device_index})")
    _pa.terminate()

    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            output_device_index=output_device_index,
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    confidence=0.7,   # more sensitive to catch user voice during bot speech
                    start_secs=0.2,   # fast interruption detection
                    stop_secs=0.8,    # triggers SmartTurn analysis after 0.8s of silence
                    min_volume=0.4,   # low enough to detect user voice reliably
                )
            ),
        )
    )

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    llm = OpenAILLMService(model=model)

    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        voice_id="TX3LPaxmHKxFdv7VOQHJ",
    )

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Keep responses under 2 sentences.",
        }
    ]

    context = OpenAILLMContext(messages=messages, tools=TOOLS)

    # SmartTurn uses an on-device ONNX model (smart-turn-v3) to semantically
    # determine when the user has truly finished speaking. It scores each
    # pause as COMPLETE or INCOMPLETE, so thinking pauses like
    # "I want to order... hmm... maybe a pizza" are held open until the
    # utterance is genuinely done. stop_secs is the hard fallback: if the
    # model keeps returning INCOMPLETE for longer than this, the turn ends anyway.
    smart_turn = LocalSmartTurnAnalyzerV3(
        params=SmartTurnParams(
            stop_secs=3.0,      # fallback: force end-of-turn after 3s of silence
            pre_speech_ms=500,  # include 500ms before speech for model context
        )
    )

    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=smart_turn)]
            )
        ),
    )

    llm.register_function("get_current_time", handle_get_current_time)
    llm.register_function("tell_joke", handle_tell_joke)
    llm.register_function("lookup_order", handle_lookup_order)

    latency_tracker = LatencyTracker(model_name=model)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            latency_tracker,      # sits between TTS and playback
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            interruption_strategies=[MinWordsInterruptionStrategy(min_words=1)],
        ),
        idle_timeout_secs=600,
    )

    print(f"Voice bot ready  [model={model}]  — speak into your microphone, Ctrl+C to stop.\n")

    runner = PipelineRunner()
    try:
        await runner.run(task)
    finally:
        latency_tracker.print_summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice bot with latency tracking")
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        choices=["gpt-4o-mini", "gpt-4o"],
        help="OpenAI model to use (default: gpt-4o-mini)",
    )
    args = parser.parse_args()
    asyncio.run(main(model=args.model))
