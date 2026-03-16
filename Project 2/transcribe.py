import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from deepgram import DeepgramClient

load_dotenv()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
if not DEEPGRAM_API_KEY:
    print("Error: DEEPGRAM_API_KEY not found in .env file")
    sys.exit(1)

if len(sys.argv) < 2:
    print("Usage: python transcribe.py <audio_file_path>")
    sys.exit(1)

audio_path = Path(sys.argv[1])
if not audio_path.exists():
    print(f"Error: File '{audio_path}' not found")
    sys.exit(1)

print(f"Transcribing: {audio_path.name}")

client = DeepgramClient(api_key=DEEPGRAM_API_KEY)

with open(audio_path, "rb") as f:
    audio_data = f.read()

response = client.listen.v1.media.transcribe_file(
    request=audio_data,
    model="nova-2",
    language="en",
    punctuate=True,
    utterances=True,
)

result = response.results
transcript = result.channels[0].alternatives[0].transcript
words = result.channels[0].alternatives[0].words

print("\n--- TRANSCRIPT ---")
print(transcript)
print("\n--- WORD-LEVEL TIMESTAMPS ---")
for word in words:
    print(f"  [{word.start:.2f}s - {word.end:.2f}s]  {word.word}")

output_path = audio_path.with_suffix(".txt")
with open(output_path, "w") as f:
    f.write("TRANSCRIPT\n")
    f.write("==========\n")
    f.write(transcript + "\n\n")
    f.write("WORD-LEVEL TIMESTAMPS\n")
    f.write("=====================\n")
    for word in words:
        f.write(f"[{word.start:.2f}s - {word.end:.2f}s]  {word.word}\n")

print(f"\nTranscript saved to: {output_path}")
