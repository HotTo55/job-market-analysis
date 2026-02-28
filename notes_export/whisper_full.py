import os
import yt_dlp
import whisper
from openai import OpenAI

URL = "https://www.youtube.com/watch?v=b0N0ufRKjKQ"
AUDIO_PATH = "C:/Users/sberry5/Documents/teaching/UDA/voice_inflection.m4a"
SUMMARY_SPEECH_PATH = "C:/Users/sberry5/Documents/teaching/UDA/summary.mp3"

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in your environment.")

ydl_opts = {
    "format": "m4a/bestaudio/best",
    "outtmpl": "C:/Users/sberry5/Documents/teaching/UDA/voice_inflection.%(ext)s",
    "postprocessors": [
        {
            "key": "FFmpegExtractAudio",
            "preferredcodec": "m4a",
        }
    ],
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([URL]) 

model = whisper.load_model("medium")
audio = whisper.load_audio(AUDIO_PATH)
audio = whisper.pad_or_trim(audio)
result = model.transcribe(audio)

transcript = result["text"]
print("Transcript:\n", transcript)

client = OpenAI(api_key=api_key)

response = client.chat.completions.create(
    model="gpt-5-mini",
    temperature=1,
    messages=[
        {"role": "system", "content": "You are a concise summarizer."},
        {
            "role": "user",
            "content": f"Summarize this transcript in 5 bullet points:\n\n{transcript}",
        },
    ],
)

summary = response.choices[0].message.content
print("\nSummary:\n", summary)

tts_response = client.audio.speech.create(
    model="gpt-4o-mini-tts",
    voice="marin",
    input=summary,
)
tts_response.write_to_file(SUMMARY_SPEECH_PATH)
print(f"\nTTS saved to: {SUMMARY_SPEECH_PATH}")