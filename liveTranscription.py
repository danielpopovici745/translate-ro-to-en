import pyaudio
import whisper
import threading
import queue
import time
import numpy as np  # Import NumPy
import lmstudio as lms


def transcribe_live(audio_queue, model):
    """Transcribes audio chunks from the queue using Whisper."""
    try:
        while True:
            data, timestamp = audio_queue.get()  # Wait for data to be available
            if data is None:  # Signal to stop transcription
                break

            # Convert bytes to NumPy array and then to float32
            audio_np = np.frombuffer(data, dtype=np.int16)
            audio_float = audio_np.astype(np.float32) / 32768.0 # Normalize to -1 to 1

            result = model.transcribe(audio=audio_float, fp16=False, language="ro") #fp16=False avoids CUDA issues on some systems
            text = result["text"]
            print(f"{text}",end="\n")  # Print with timestamp
            translate(text)
    except Exception as e:
        print(f"Transcription error: {e}")


def record_audio(audio_queue, duration=4):
    """Records audio from the microphone and puts it into the queue."""
    try:
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                         channels=1,
                         rate=16000,  # Whisper requires 16kHz - HARDCODED VALUE
                         input=True)

        print("Recording...")
        start_time = time.time()


        while True:
            data = stream.read(int(duration * 16000))  # Read a chunk of audio
            timestamp = time.time() - start_time #Corrected timestamp calculation
            audio_queue.put((data, timestamp))



    except Exception as e:
        print(f"Recording error: {e}")
    finally:
        print("Stopping Recording")
        if 'stream' in locals():  # Check if stream was initialized
            stream.stop_stream()
            stream.close()
        if 'p' in locals(): #check if pyaudio instance exists
            p.terminate()

        audio_queue.put((None, None))  # Signal the transcription thread to stop

def translate(text):
  model = lms.llm()
  for fragment in model.respond_stream("You are a professional translator. Translate the following Romanian text to English accurately and naturally. Don't say anything more. If there is nothing to translate or you can't translate don't say anything." + text): 
      print(fragment.content, end="", flush=True)  
  print()

def main():
    """Main function to orchestrate recording and transcription."""
    model = whisper.load_model("large-v3-turbo")  # Choose a Whisper model (tiny, base, small, medium, large)
    audio_queue = queue.Queue()

    # Start the transcription thread
    transcription_thread = threading.Thread(target=transcribe_live, args=(audio_queue, model))
    transcription_thread.daemon = True  # Allow main thread to exit even if this is running
    transcription_thread.start()

    # Start the recording thread
    recording_thread = threading.Thread(target=record_audio, args=(audio_queue,))
    recording_thread.daemon = True
    recording_thread.start()


    try:
        while True:  # Keep the main thread running indefinitely
            time.sleep(1) # Check every second if a keyboard interrupt is received
    except KeyboardInterrupt:
        print("Interrupted by user.")

    # Ensure the transcription thread exits gracefully
    audio_queue.put((None, None))  # Signal the transcription thread to stop
    transcription_thread.join() #wait for transcription thread to exit


if __name__ == "__main__":
    main()