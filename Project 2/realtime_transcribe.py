import sys
import os
import threading
import queue
from dotenv import load_dotenv
import sounddevice as sd
from deepgram import DeepgramClient
from deepgram.listen.v1.types.listen_v1results import ListenV1Results

SAMPLE_RATE = 16000
CHUNK_MS = 100
CHUNK_FRAMES = int(SAMPLE_RATE * CHUNK_MS / 1000)

load_dotenv()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
if not DEEPGRAM_API_KEY:
    print("Error: DEEPGRAM_API_KEY not found in .env file")
    sys.exit(1)

audio_queue = queue.Queue()
stop_event = threading.Event()

def audio_callback(indata, frames, time, status):
    if not stop_event.is_set():
        audio_queue.put(bytes(indata))

client = DeepgramClient(api_key=DEEPGRAM_API_KEY)

print("Connecting to Deepgram...")

with client.listen.v1.connect(
    model="nova-2",
    encoding="linear16",
    sample_rate=str(SAMPLE_RATE),
    punctuate="true",
    interim_results="true",
) as socket:

    def send_audio():
        while not stop_event.is_set():
            try:
                chunk = audio_queue.get(timeout=0.1)
                socket.send_media(chunk)
            except queue.Empty:
                continue

    sender = threading.Thread(target=send_audio, daemon=True)
    sender.start()

    stream = sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=CHUNK_FRAMES,
        callback=audio_callback,
    )

    print("Listening... Press Ctrl+C to stop.\n")
    stream.start()

    try:
        for message in socket:
            if isinstance(message, ListenV1Results):
                transcript = message.channel.alternatives[0].transcript
                if transcript:
                    if message.is_final:
                        print(f"\r{transcript}                    ")
                    else:
                        print(f"\r[...] {transcript}", end="", flush=True)
    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        stop_event.set()
        stream.stop()
        stream.close()
        try:
            socket.send_close_stream()
        except Exception:
            pass
        print("Done.")
