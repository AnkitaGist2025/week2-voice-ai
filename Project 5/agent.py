"""
Day 5 – LiveKit Agents voice pipeline
======================================
Pipeline: SileroVAD → Deepgram STT → OpenAI GPT-4.1-mini → ElevenLabs TTS

The agent joins a LiveKit room as a participant and responds to anyone who speaks.

Run (development mode — hot reload, local room simulation):
    python agent.py dev

Run (connect to a real LiveKit server):
    python agent.py start
"""

import os

from dotenv import load_dotenv
from livekit.agents import Agent, AgentSession, AutoSubscribe, JobContext, WorkerOptions, cli
from livekit.plugins import deepgram, elevenlabs, openai, silero

load_dotenv()


# ── Prewarm ───────────────────────────────────────────────────────────────────
# prewarm_fnc runs once per worker process before any job is assigned.
# Loading the Silero ONNX model here (blocking) means it's ready in memory
# when the first call arrives — no cold-start latency for VAD.

def prewarm(proc):
    proc.userdata["vad"] = silero.VAD.load()


# ── Entrypoint ────────────────────────────────────────────────────────────────
# Called once per room job. ctx.room is the LiveKit Room the agent joined.

async def entrypoint(ctx: JobContext):
    # Connect to the LiveKit room and subscribe to all audio/video tracks
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    session = AgentSession(
        # VAD: on-device Silero model — detects end-of-speech before sending to STT
        vad=ctx.proc.userdata["vad"],

        # STT: Deepgram nova-3 — low-latency streaming transcription
        stt=deepgram.STT(api_key=os.getenv("DEEPGRAM_API_KEY")),

        # LLM: GPT-4.1-mini — fast, cost-effective responses
        llm=openai.LLM(
            model="gpt-4.1-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
        ),

        # TTS: ElevenLabs turbo — low-latency streaming synthesis
        tts=elevenlabs.TTS(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id=os.getenv("ELEVENLABS_VOICE_ID", "TX3LPaxmHKxFdv7VOQHJ"),
        ),

        # Interrupt the agent if the user starts speaking mid-response
        allow_interruptions=True,
    )

    agent = Agent(
        instructions="You are a helpful assistant. Keep responses brief.",
    )

    # Start the session — wires up audio I/O to the room and begins listening
    await session.start(agent, room=ctx.room)


# ── Worker entry ──────────────────────────────────────────────────────────────
# WorkerOptions picks up LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET
# from the environment automatically.

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        )
    )
