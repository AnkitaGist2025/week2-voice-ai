import sys
import os
import io
import wave
import time
from dotenv import load_dotenv
import sounddevice as sd
import numpy as np
from deepgram import DeepgramClient
from elevenlabs import ElevenLabs
from openai import OpenAI

RECORD_SECONDS = 5
SAMPLE_RATE = 16000
DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"
TTS_SAMPLE_RATE = 24000

load_dotenv()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

missing = [k for k, v in {
    "DEEPGRAM_API_KEY": DEEPGRAM_API_KEY,
    "ELEVENLABS_API_KEY": ELEVENLABS_API_KEY,
    "OPENAI_API_KEY": OPENAI_API_KEY,
}.items() if not v]
if missing:
    print(f"Error: missing keys in .env: {', '.join(missing)}")
    sys.exit(1)

deepgram = DeepgramClient(api_key=DEEPGRAM_API_KEY)
elevenlabs = ElevenLabs(api_key=ELEVENLABS_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

pipeline_start = time.time()

# ── Step 1: Record microphone ────────────────────────────────────────────────
print(f"\n🎙  Recording for {RECORD_SECONDS} seconds... speak now!")
recording = sd.rec(
    frames=RECORD_SECONDS * SAMPLE_RATE,
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="int16",
)
sd.wait()
print("Recording done.")
record_end = time.time()

# ── Step 2: Transcribe with Deepgram ────────────────────────────────────────
print("\n📝 Transcribing...")
t0 = time.time()

buf = io.BytesIO()
with wave.open(buf, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)  # 16-bit = 2 bytes
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(recording.tobytes())
wav_bytes = buf.getvalue()

response = deepgram.listen.v1.media.transcribe_file(
    request=wav_bytes,
    model="nova-2",
    language="en",
    punctuate=True,
)
transcript = response.results.channels[0].alternatives[0].transcript
stt_time = time.time() - t0

if not transcript:
    print("No speech detected. Please try again.")
    sys.exit(0)

print(f"You said: \"{transcript}\"  ({stt_time:.2f}s)")

# ── Step 3: Get response from OpenAI ────────────────────────────────────────
print("\n🤖 Getting AI response...")
t0 = time.time()

completion = openai_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant. Keep responses concise — 1 to 2 sentences."},
        {"role": "user", "content": transcript},
    ],
)
ai_response = completion.choices[0].message.content
llm_time = time.time() - t0

print(f"AI said: \"{ai_response}\"  ({llm_time:.2f}s)")

# ── Step 4: Generate & play speech with ElevenLabs ──────────────────────────
print("\n🔊 Generating and playing speech...")
t0 = time.time()
first_chunk_time = None

audio_stream = elevenlabs.text_to_speech.stream(
    voice_id=DEFAULT_VOICE_ID,
    text=ai_response,
    model_id="eleven_turbo_v2_5",
    output_format="pcm_24000",
)

with sd.RawOutputStream(samplerate=TTS_SAMPLE_RATE, channels=1, dtype="int16") as player:
    for chunk in audio_stream:
        if chunk:
            if first_chunk_time is None:
                first_chunk_time = time.time()
            player.write(chunk)

tts_time = time.time() - t0

# ── Summary ──────────────────────────────────────────────────────────────────
total_time = time.time() - pipeline_start
print(f"""
──────────────────────────────
 Latency breakdown
──────────────────────────────
 Recording:      {RECORD_SECONDS:.1f}s (fixed)
 STT (Deepgram): {stt_time:.2f}s
 LLM (OpenAI):   {llm_time:.2f}s
 TTS (ElevenLabs): {tts_time:.2f}s
 ─────────────────────────────
 Total:          {total_time:.2f}s
──────────────────────────────
""")
