import pyaudio
import wave
import keyboard
import time
import os
import whisper
import lmstudio as lms

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
  model = lms.llm()
  for fragment in model.respond_stream("Translate the Romanian to English. Don't say anything more." + text):  
      print(fragment.content, end="", flush=True)  
  print()


recorded_file = record_audio()
if recorded_file:
    transcription = transcribe(recorded_file)
    if transcription:
      print("Transcription:"+ transcription, end="\n \n")
      translate(transcription)  