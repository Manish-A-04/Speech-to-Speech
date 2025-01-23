import pyaudio
import wave
import numpy as np
import whisper
import time
import requests
import json


THRESHOLD = 500 
PAUSE_DURATION = 2  
SAMPLE_RATE = 16000
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
OUTPUT_WAVE_FILENAME = "recorded_audio.wav"
OLLAMA_URL = "http://localhost:11434/api/generate"  


def record_audio():
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=SAMPLE_RATE, input=True,
                        frames_per_buffer=CHUNK)

    print("Listening... Speak now.")
    frames = []
    silent_chunks = 0
    while True:
        data = stream.read(CHUNK)
        frames.append(data)

        
        audio_data = np.frombuffer(data, dtype=np.int16)
        if np.max(audio_data) < THRESHOLD:
            silent_chunks += 1
        else:
            silent_chunks = 0

        
        if silent_chunks > (SAMPLE_RATE / CHUNK * PAUSE_DURATION):
            print("Finished recording.")
            break

    stream.stop_stream()
    stream.close()
    audio.terminate()

    
    with wave.open(OUTPUT_WAVE_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b''.join(frames))

    return OUTPUT_WAVE_FILENAME


def transcribe_audio(audio_file):
    print("Transcribing audio...")
    model = whisper.load_model("base") 
    result = model.transcribe(audio_file)
    return result['text']


def query_ollama(prompt):
    print("Querying Ollama model...")
    headers = {'Content-Type': 'application/json'}
    data = {
        "prompt": prompt,
        "model": "qwen2.5-coder:latest"  
    }
    
    
    response = requests.post(OLLAMA_URL, json=data, headers=headers, stream=True)
    
    
    complete_response = ""
    
    if response.status_code == 200:
        try:
            
            for line in response.iter_lines():
                if line:  
                    line_data = json.loads(line.decode('utf-8'))  
                    complete_response += line_data.get("response", "")  
            
            return complete_response.strip()  
        except Exception as e:
            return f"Error processing response: {e}\nRaw response: {response.text}"
    else:
        return f"Error: {response.status_code} - {response.text}"

def main():
    audio_file = record_audio()
    transcription = transcribe_audio(audio_file)
    print(f"Transcription: {transcription}")

    if transcription.strip():
        response = query_ollama(transcription)
        print(f"Ollama response: {response}")
    else:
        print("No speech detected.")

if __name__ == "__main__":
    main()
