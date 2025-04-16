import pyaudio
import wave
import keyboard
import time
import os
import whisper
import requests

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
OUTPUT_FILENAME = "recordedFile.wav"

def record_audio():

  audio = pyaudio.PyAudio()
  stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

  frames = []
  print("Press SPACE to start recording!")
  keyboard.wait('space')
  print("Recording... Press SPACE to stop.") 
  time.sleep(0.2)
  while True:  
    try:
      data = stream.read(CHUNK)
      frames.append(data)
    except KeyboardInterrupt:
      break
    if keyboard.is_pressed('space'):
      print("Stopping recording after a brief delay")
      time.sleep(0.2)
      break
  stream.stop_stream()
  stream.close()
  audio.terminate()

  waveFile= wave.open(OUTPUT_FILENAME, 'wb')
  waveFile.setnchannels(CHANNELS)
  waveFile.setsampwidth(audio.get_sample_size(FORMAT))
  waveFile.setframerate(RATE) 
  waveFile.writeframes(b''.join(frames))
  waveFile.close()

  return os.path.abspath(OUTPUT_FILENAME)  
  

def transcribe(filePath):
  if not os.path.exists(filePath):
    print(f"Error: File not found at {filePath}")
    return None
  try:
    model = whisper.load_model("base")
    result = model.transcribe(filePath, language="ro")
    return result["text"]
  except Exception as e:  
    print(f"Error transcribing audio: {e}")
    return None 

def translate(text):
  try:
    headers = {"Content-Type": "application/json"}
    payload = {
      "model": "gemma-3-12b-it",
      "messages": [
        {"role": "system", "content": "You are a professional Romanian to English translator. Only translate the words, nothing more."},
        {"role": "user", "content": text}  # User's prompt
      ],
      "temperature":0.1,
      "stream": False
    }

    response = requests.post(f"http://127.0.0.1:1234/v1/chat/completions", headers=headers, json=payload)
    response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

    translation = response.json()["choices"][0]["message"]["content"]  # Extract the translation from the JSON response
    return translation

  except requests.exceptions.RequestException as e:
        print(f"Error translating with LM Studio: {e}")
        return None
  except (KeyError, IndexError) as e:
      print(f"Error parsing LM Studio response: {e}.  Check the model and endpoint.")
      return None

recorded_file = record_audio()
if recorded_file:
    transcription = transcribe(recorded_file)
    if transcription:
      print("Transcription:"+ transcription)
      translation = translate(transcription)
      if translation:
         print("Translation: " + translation)