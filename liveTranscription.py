import pyaudio
import whisper
import queue
import time
import numpy as np
import lmstudio as lms
import os
import pygame
from concurrent.futures import ThreadPoolExecutor
from google.cloud import texttospeech
import threading
import queue


def is_silent(audio_chunk, threshold=43):
    """Determines if the audio chunk is silent based on its energy."""
    audio_np = np.frombuffer(audio_chunk, dtype=np.int16)
    energy = np.sqrt(np.mean(audio_np**2))
    return energy < threshold


def transcribe_live(audio_queue, model, translator_model, translation_queue, stop_event=None):
    """Transcribes audio chunks from the queue using Whisper and sends text to the translation queue."""
    try:
        batch = []
        batch_size = 5  # Number of chunks to batch process
        while not stop_event.is_set():  # Check stop_event
            data = audio_queue.get()
            if data is None:  # Signal to stop transcription
                break

            # Skip silent chunks
            if is_silent(data):
                continue

            batch.append(data)
            if len(batch) >= batch_size:
                # Process the batch
                audio_np = np.concatenate([np.frombuffer(chunk, dtype=np.int16) for chunk in batch])
                audio_float = audio_np.astype(np.float32) / 32768.0

                # Transcribe audio
                result = model.transcribe(audio=audio_float, fp16=False, language="ro")
                text = result.get("text", "").strip()
                if text:
                    print(f"\nTranscribed: {text}\n")
                    translation_queue.put(text)  # Send text to translation queue

                batch = []  # Clear the batch

        # Process any remaining audio in the batch
        if batch:
            audio_np = np.concatenate([np.frombuffer(chunk, dtype=np.int16) for chunk in batch])
            audio_float = audio_np.astype(np.float32) / 32768.0
            result = model.transcribe(audio=audio_float, fp16=False, language="ro")
            text = result.get("text", "").strip()
            if text:
                print(f"\nTranscribed: {text}\n")
                translation_queue.put(text)

    except Exception as e:
        print(f"Transcription error: {e}")


def record_audio(audio_queue, duration=2.0, stop_event=None, input_device=None):
    """Records audio from the microphone and puts it into the queue."""
    RATE = 16000
    CHUNK = int(duration * RATE)

    try:
        p = pyaudio.PyAudio()

        # Find the index of the selected input device
        input_device_index = None
        for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)
            if device_info["name"] == input_device:
                input_device_index = i
                break

        if input_device_index is None:
            raise ValueError(f"Input device '{input_device}' not found.")

        stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, input_device_index=input_device_index)

        print("Recording...")

        while not stop_event.is_set():  # Check stop_event
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_queue.put(data)
    except Exception as e:
        print(f"Recording error: {e}")
    finally:
        if 'stream' in locals():
            stream.stop_stream()
            stream.close()
        if 'p' in locals():
            p.terminate()

        audio_queue.put(None)  # Signal the transcription thread to stop


def translate(text, model):
    """Translates Romanian text to English using the lmstudio model."""
    translated_text = ""
    for fragment in model.respond_stream(
        "You are a professional translator. Translate the following Romanian text to English accurately and naturally. Don't say anything more. If there is nothing to translate or you can't translate don't say anything." + text
    ):
        translated_text += fragment.content
    return translated_text


def translation_worker(translation_queue, translator_model, tts_queue, stop_event=None):
    """Worker function to translate text asynchronously and send it to the TTS queue."""
    while not stop_event.is_set():
        text = translation_queue.get()
        if text is None:  # Signal to stop translation
            break
        try:
            translated_text = translate(text, translator_model)
            if translated_text:
                print(f"Translated: {translated_text}\n")
                tts_queue.put(translated_text)  # Send to TTS queue
        except Exception as e:
            print(f"Translation error: {e}")


def tts_worker(tts_queue, stop_event=None, output_device=None):
    """Worker function for TTS to process translations asynchronously using Google Cloud Text-to-Speech."""
    client = texttospeech.TextToSpeechClient()

    while not stop_event.is_set():
        text = tts_queue.get()
        if text is None:  # Signal to stop TTS
            break
        try:
            # Set up the input text
            synthesis_input = texttospeech.SynthesisInput(text=text)

            # Configure the voice request
            voice = texttospeech.VoiceSelectionParams(
                language_code="en-US",
                name="en-US-Standard-D",  # Specify the desired voice
                ssml_gender=texttospeech.SsmlVoiceGender.MALE
            )

            # Configure the audio output
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16  # Use WAV format
            )

            # Perform the text-to-speech request
            response = client.synthesize_speech(
                input=synthesis_input, voice=voice, audio_config=audio_config
            )

            # Save the audio to a file
            temp_audio_file = os.path.join(os.getcwd(), "temp_audio.wav")
            with open(temp_audio_file, "wb") as out:
                out.write(response.audio_content)

            # Initialize pygame mixer with the selected output device
            if output_device:
                pygame.mixer.init(devicename=output_device)
            else:
                pygame.mixer.init()  # Use the default output device if none is specified

            pygame.mixer.music.load(temp_audio_file)
            pygame.mixer.music.play()

            # Wait for the audio to finish playing
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)

            # Stop the mixer and clean up
            pygame.mixer.music.stop()
            pygame.mixer.quit()

            # Remove the temporary audio file
            os.remove(temp_audio_file)
        except Exception as e:
            print(f"Error in TTS: {e}")


def main(stop_event=None, input_device=None, output_device=None):
    """Main function to orchestrate recording, transcription, translation, and TTS."""
    try:
        # Load models
        print("Loading Whisper model...")
        transcription_model = whisper.load_model("large-v3-turbo")
        print("Loading translation model...")
        translator_model = lms.llm()

        # Create queues for audio, translation, and TTS
        audio_queue = queue.Queue()
        translation_queue = queue.Queue()
        tts_queue = queue.Queue()

        # Use ThreadPoolExecutor to manage threads
        with ThreadPoolExecutor(max_workers=4) as executor:
            executor.submit(record_audio, audio_queue, stop_event=stop_event, input_device=input_device)
            executor.submit(transcribe_live, audio_queue, transcription_model, translator_model, translation_queue, stop_event=stop_event)
            executor.submit(translation_worker, translation_queue, translator_model, tts_queue, stop_event=stop_event)
            executor.submit(tts_worker, tts_queue, stop_event=stop_event, output_device=output_device)

            print(f"Using output device: {output_device}")
            print("Press Ctrl+C to stop.")
            while not (stop_event and stop_event.is_set()):  # Check the stop event
                time.sleep(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting...")
    finally:
        # Ensure threads exit gracefully
        audio_queue.put(None)  # Signal the transcription thread to stop
        translation_queue.put(None)  # Signal the translation thread to stop
        tts_queue.put(None)  # Signal the TTS thread to stop


if __name__ == "__main__":
    main()