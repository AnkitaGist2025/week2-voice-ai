import sys
import os
import time
from dotenv import load_dotenv
from elevenlabs import ElevenLabs
import sounddevice as sd

DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"

# PCM format: 24kHz sample rate, 16-bit mono — raw audio, no decoding needed
SAMPLE_RATE = 24000

load_dotenv()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
if not ELEVENLABS_API_KEY:
    print("Error: ELEVENLABS_API_KEY not found in .env file")
    sys.exit(1)

if len(sys.argv) < 2:
    print("Usage: python streaming_tts.py \"Your text here\"")
    sys.exit(1)

text = " ".join(sys.argv[1:])
print(f"Text: \"{text}\"")
print("Requesting audio stream from ElevenLabs...")

client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

start_time = time.time()
first_chunk_time = None

audio_stream = client.text_to_speech.stream(
    voice_id=DEFAULT_VOICE_ID,
    text=text,
    model_id="eleven_turbo_v2_5",
    output_format="pcm_24000",
)

with sd.RawOutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16") as player:
    for chunk in audio_stream:
        if chunk:
            if first_chunk_time is None:
                first_chunk_time = time.time()
                ttfa = first_chunk_time - start_time
                print(f"Time-to-first-audio: {ttfa:.2f}s  ← playback starts here")
            player.write(chunk)

total_time = time.time() - start_time
print(f"Total time:          {total_time:.2f}s")
print(f"Streaming advantage: audio started {total_time - ttfa:.2f}s before full generation finished")
