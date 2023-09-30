import os

import whisper
import sounddevice as sd
import numpy as np
import wave
import threading
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

SAMPLE_RATE = 44100
CHANNELS = 1
DATA_TYPE = np.int16
RAW_FILENAME = "recorded_audio.wav"
TRANSCRIPTION_FILENAME = "transcription.text"

audio_buffer = []


def recording_callback(indata, frames, time, status):
    audio_buffer.append(indata.copy())


def record_audio():
    # This creates a streaming object that uses our callback.
    # It'll keep recording until we stop it.
    with sd.InputStream(callback=recording_callback, channels=CHANNELS, samplerate=SAMPLE_RATE, dtype=DATA_TYPE):
        while not stop_recording:
            sd.sleep(1000)  # Sleep for 1 second increments and then check for the stop flag

    global raw_audio_data
    raw_audio_data = np.concatenate(audio_buffer, axis=0)


def save_audio(raw_audio_data, filename):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 2 bytes for int16
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(raw_audio_data.tobytes())

    print(f"Saved to {filename}")

    return


def transcribe_audio(raw_audio_file, transcription_file):
    model = whisper.load_model("base")
    audio = raw_audio_file
    result = model.transcribe(audio)

    with open(transcription_file, "w", encoding="utf-8") as txt:
        txt.write(result["text"])

    return result["text"]

def chunk_text(raw_text):
    chunked_text = [raw_text]
    return chunked_text


def ask_questions_of_chunk(prelude, prompt_list, prompts, chunk):
    aggregate_response = ''
    for prompt in prompt_list:

        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[
                {"role": "system", "content": prelude + '\r\r\"' + chunk},
                {"role": "user", "content": prompts[prompt]},
            ],
            temperature=0.9
        )

        chat_response = response.choices[0]['message']['content'] + '\r'
        aggregate_response += prompt + '\r' + chat_response + '\r'

    return aggregate_response


def main():
    prelude = 'the following is a transcript between an interviewer and an entrepreneur, who you should call "they"\r'\
        + 'who is starting a business and discussing their business and their product\r'\
        + 'please answer as a helpful ai agent, only using information in the supplied text transcript'\
        + 'of the interview.\r'\
        + 'please be as detailed as possible. if you don\'t know the answer, please answer "unknown".\r'

    prompt_list = ["NAME", "PROBLEM", "SOLUTION", "TEAM", "TRACTION", "TECH", "TAM", "TIMING", "LEISURE", "TEAM EXPERIENCE", "FIRST TIME FOUNDER?"]

    prompts = {
                "NAME": 'what is the name of the company',
                "PROBLEM": 'what problems are they solving, and what customers have these problems',
                "SOLUTION": 'how does their product solve the problem, in as much detail as possible',
                "TEAM": 'who are the founders of the company and what are their educations and roles',
                "TRACTION": 'how many customers do they have, and what are the names of their customers and prospects, including those on their waitlist',
                "TECH": 'what technologies are they using in their product and what makes those technologies unique',
                "TAM": 'how big is the market they\'re addressing both in numbers of customers and dollar size',
                "TIMING": 'is there something happening in technology or the market or society that makes this more relevant or more possible right now',
                "LEISURE": 'what do the founders and cofounders do in their spare time for hobbies, avocations and interests, sports',
                "TEAM EXPERIENCE": 'is this the first time the founders have worked together or do they have prior experience together',
                "FIRST TIME FOUNDER?": 'has the ceo and other members of the founding team started another startup previously or is this their first company'
            }
    """
    global stop_recording
    stop_recording = False

    recording_thread = threading.Thread(target=record_audio)
    recording_thread.start()

    input("Press Enter to stop recording...")
    stop_recording = True
    recording_thread.join()

    save_audio(raw_audio_data, RAW_FILENAME)
    """

    raw_text = transcribe_audio(RAW_FILENAME, TRANSCRIPTION_FILENAME)
    chunked_text = chunk_text(raw_text)

    for chunk in chunked_text:
        chunk_answers = ask_questions_of_chunk(prelude, prompt_list, prompts, chunk)
        pass

if __name__ == "__main__":
    main()