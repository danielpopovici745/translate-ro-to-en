import pyaudio
import whisper
import threading
import queue
import time
import numpy as np
import lmstudio as lms
import os


def transcribe_live(audio_queue, model, translator_model):
    """Transcribes audio chunks from the queue using Whisper and translates them."""
    try:
        while True:
            data = audio_queue.get()  # Wait for data to be available
            if data is None:  # Signal to stop transcription
                break

            # Convert bytes to NumPy array and normalize to float32
            audio_np = np.frombuffer(data, dtype=np.int16)
            audio_float = audio_np.astype(np.float32) / 32768.0

            # Transcribe audio
            result = model.transcribe(audio=audio_float, fp16=False, language="ro")
            text = result.get("text", "").strip()
            if text:
                print(f"\nTranscribed: {text}\n")
                translated_text = translate(text, translator_model)
                if translated_text:
                    print(f"Translated: {translated_text}\n")

            # Print a separator line
            terminal_width = os.get_terminal_size().columns
            print('_' * terminal_width)
    except Exception as e:
        print(f"Transcription error: {e}")


def record_audio(audio_queue, duration=5):
    """Records audio from the microphone and puts it into the queue."""
    RATE = 16000

    try:
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True)

        print("Recording...")

        while True:
            data = stream.read(int(duration * RATE), exception_on_overflow=False)  # Read a chunk of audio
            audio_queue.put(data)
    except Exception as e:
        print(f"Recording error: {e}")
    finally:
        print("Stopping Recording")
        if 'stream' in locals():
            stream.stop_stream()
            stream.close()
        if 'p' in locals():
            p.terminate()

        audio_queue.put(None)  # Signal the transcription thread to stop


def translate(text, model):
    """Translates Romanian text to English using the lmstudio model."""

    for fragment in model.respond_stream("You are a professional translator. Translate the following Romanian text to English accurately and naturally. Don't say anything more. If there is nothing to translate or you can't translate don't say anything." + text): 
      print(fragment.content, end="", flush=True)
    print()


def main():
    """Main function to orchestrate recording, transcription, and translation."""
    try:
        # Load models
        print("Loading Whisper model...")
        transcription_model = whisper.load_model("large-v3-turbo")
        print("Loading translation model...")
        translator_model = lms.llm()

        # Create a queue for audio data
        audio_queue = queue.Queue()

        # Start the transcription thread
        transcription_thread = threading.Thread(
            target=transcribe_live, args=(audio_queue, transcription_model, translator_model)
        )
        transcription_thread.daemon = True
        transcription_thread.start()

        # Start the recording thread
        recording_thread = threading.Thread(target=record_audio, args=(audio_queue,))
        recording_thread.daemon = True
        recording_thread.start()

        # Keep the main thread running
        print("Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting...")
    finally:
        # Ensure threads exit gracefully
        audio_queue.put(None)  # Signal the transcription thread to stop
        transcription_thread.join()
        print("Transcription thread stopped.")

main()