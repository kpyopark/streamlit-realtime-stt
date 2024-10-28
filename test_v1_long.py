import os
from dotenv import load_dotenv

from google.cloud import speech_v1 as speech

from pydub import AudioSegment
import io
import asyncio
import wave

load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
TEST_FILE_NAME = os.getenv("TEST_FILE_NAME")

# Audio recording parameters
STREAMING_LIMIT = 240000  # 4 minutes
SAMPLE_RATE = 8000
DURATION_MS = 1000 # ms
CHUNK_SIZE = int(SAMPLE_RATE * 1000 / DURATION_MS)  # 1 sec

def convert_to_wav(audio_file):
    audio = AudioSegment.from_file(audio_file)
    audio = audio.set_channels(1)  # mono
    audio = audio.set_frame_rate(SAMPLE_RATE)  # 16kHz
    
    # WAV 데이터를 메모리에 저장
    wav_buffer = io.BytesIO()
    audio.export(wav_buffer, format="wav")
    #wav_data = wav_buffer.getvalue()
    return wav_buffer

def split_wav(wav_data_bytes, need_header: bool = False):
    chunks = []
    with wave.open(wav_data_bytes, 'rb') as wav_data:
        params = wav_data.getparams()
        frames_per_chunk = CHUNK_SIZE // (params.sampwidth * params.nchannels)
        is_first_segment = True
        while True:
            frames = wav_data.readframes(frames_per_chunk)
            if not frames:
                break
            chunk = io.BytesIO()
            with wave.open(chunk, 'wb') as chunk_wav:
                # if need_header or is_first_segment:
                #     is_first_segment = False
                chunk_wav.setparams(params)
                chunk_wav.writeframes(frames)
            chunks.append(chunk.getvalue())
    return chunks

def transcribe_streaming(wav_data, language_code="ko-KR"):
    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE,
        language_code="cmn-Hant-TW", # "yue-Hant-HK", #"cmn-Hans-CN",
        max_alternatives=1
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=True
    )
    stream = split_wav(wav_data)
    
    requests = (
        speech.StreamingRecognizeRequest(audio_content=chunk) for chunk in stream    
    )
    
    responses = client.streaming_recognize(requests=requests, config=streaming_config)
    
    for response in responses:
        for result in response.results:
            for alternative in result.alternatives:
                print("Transcript: {}".format(alternative.transcript))
                print("Confidence: {}".format(alternative.confidence))

if __name__ == "__main__":
    wav_data = convert_to_wav(TEST_FILE_NAME)
    transcribe_streaming(wav_data)