import sys
import os
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from elevenlabs import ElevenLabs

# Default voice: Rachel (a natural-sounding ElevenLabs pre-made voice)
DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"
OUTPUT_FILE = "output.mp3"

load_dotenv()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
if not ELEVENLABS_API_KEY:
    print("Error: ELEVENLABS_API_KEY not found in .env file")
    sys.exit(1)

if len(sys.argv) < 2:
    print("Usage: python generate_speech.py \"Your text here\"")
    sys.exit(1)

text = " ".join(sys.argv[1:])
print(f"Generating speech for: \"{text}\"")

client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

audio_chunks = client.text_to_speech.convert(
    voice_id=DEFAULT_VOICE_ID,
    text=text,
    model_id="eleven_turbo_v2_5",
    output_format="mp3_44100_128",
)

with open(OUTPUT_FILE, "wb") as f:
    for chunk in audio_chunks:
        f.write(chunk)

print(f"Saved to: {OUTPUT_FILE}")
print("Playing audio...")

subprocess.run(["afplay", OUTPUT_FILE])
