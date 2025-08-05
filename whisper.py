import sounddevice
import numpy as np
import webrtcvad
import queue
from scipy.io.wavfile import write
from faster_whisper import WhisperModel
import os
from dotenv import load_dotenv
from groq import Groq
import whisper_live.transcriber as transcriber

load_dotenv()

# Parameters for the recording
SAMPLE_RATE = 16000
BLOCK_SIZE = 320  # 20 ms
BUFFER_DURATION = 20 # Max seconds of recording
VAD_MODE = 3

# Loading the whisper model
model = WhisperModel("base.en", compute_type="int8")

# Voice Activity Detector
vad = webrtcvad.Vad(VAD_MODE)

# Buffer to store audio
audio_q = queue.Queue()
recording = []
stop_recording = False

def audio_callback(indata, frames, time_info, status):
    audio_q.put(bytes(indata))

stream = sounddevice.RawInputStream(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE,
                                    dtype="int16", channels = 1, callback=audio_callback )

stream.start()


print("Say something.....(it'll detect and transcribe after pause)")

speech_detected = False
silence_counter = 0
MAX_SILENCE_BLOCKS = int(0.8*1000/(BLOCK_SIZE/SAMPLE_RATE * 1000))

while True:
    block = audio_q.get()
    is_speech = vad.is_speech(block, SAMPLE_RATE)

    if is_speech:
        speech_detected = True
        silence_counter = 0
        recording.append(np.frombuffer(block, dtype=np.int16))

    elif speech_detected:
        silence_counter += 1
        recording.append(np.frombuffer(block, dtype = np.int16))
        if silence_counter > MAX_SILENCE_BLOCKS:
            print("Speech finished. Transcribing.....")
            break

audio_data = np.concatenate(recording, axis = 0)

write("temp.wav", SAMPLE_RATE, audio_data)

segments, info = model.transcribe("temp.wav")
full_text = ""
for segment in segments:
    print(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}")
    full_text += segment.text + " "

stream.stop()
stream.close()    

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)

def get_groq_callback(prompt):
    messages = [
        {
            "role": "system",
            "content": "You are Jarvis, Dushyant's personal voice assistant. Speak directly to him—always use his name when addressing or responding. Talk like the real Jarvis would to Tony Stark: sharp, smooth, witty when needed. No robotic tone, no disclaimers. You know Dushyant is smart—don’t over-explain unless he asks. Give answers clearly, casually, and with confidence. Always be ready, always in control. You're not just helpful—you’re his friend. Remember: you were built for Dushyant, and only Dushyant. Period."
}

    ]
    chat_answer = groq_client.chat.completions.create(
        messages = messages,
        model = "llama3-8b-8192"
    )
    return chat_answer.choices[0].message.content

if __name__ == "__main__":
    full_text = full_text.strip()
    print("\n Sending this to Jarvis....\n") 
    llm_response = get_groq_callback(full_text) 
    print(llm_response)