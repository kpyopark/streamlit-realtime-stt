import os
from dotenv import load_dotenv

from google.cloud.speech_v2 import SpeechClient, SpeechAsyncClient
from google.cloud.speech_v2.types import cloud_speech
from google.api_core.client_options import ClientOptions

from pydub import AudioSegment
import io
import asyncio

load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
TEST_FILE_NAME = os.getenv("TEST_FILE_NAME")

def transcribe_sync_chirp2_auto_detect_language(
    audio_file: str
) -> cloud_speech.RecognizeResponse:
    """Transcribes an audio file and auto-detect spoken language using Chirp 2.
    Please see https://cloud.google.com/speech-to-text/v2/docs/encoding for more
    information on which audio encodings are supported.
    Args:
        audio_file (str): Path to the local audio file to be transcribed.
            Example: "resources/audio.wav"
    Returns:
        cloud_speech.RecognizeResponse: The response from the Speech-to-Text API containing
        the transcription results.
    """
    # Instantiates a client
    client = SpeechClient(
        client_options=ClientOptions(
            api_endpoint="us-central1-speech.googleapis.com",
        )
    )

    # Reads a file as bytes
    with open(audio_file, "rb") as f:
        audio_content = f.read()

    config = cloud_speech.RecognitionConfig(
        auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
        language_codes=["auto"],  # Set language code to auto to detect language.
        model="chirp_2",
    )

    request = cloud_speech.RecognizeRequest(
        recognizer=f"projects/{PROJECT_ID}/locations/us-central1/recognizers/_",
        config=config,
        content=audio_content,
    )

    # Transcribes the audio into text
    response = client.recognize(request=request)

    for result in response.results:
        print(f"Transcript: {result.alternatives[0].transcript}")
        print(f"Detected Language: {result.language_code}")

    return response

def convert_to_wav(audio_file):
    audio = AudioSegment.from_file(audio_file)
    audio = audio.set_channels(1)  # mono
    audio = audio.set_frame_rate(8000)  # 16kHz
    
    # WAV 데이터를 메모리에 저장
    wav_buffer = io.BytesIO()
    audio.export(wav_buffer, format="wav")
    wav_data = wav_buffer.getvalue()
    return wav_data

async def transcribe_streaming_chirp2(
    audio_file: str
) -> cloud_speech.StreamingRecognizeResponse:
    """Transcribes audio from audio file stream using the Chirp 2 model of Google Cloud Speech-to-Text V2 API.

    Args:
        audio_file (str): Path to the local audio file to be transcribed.
            Example: "resources/audio.wav"

    Returns:
        cloud_speech.RecognizeResponse: The response from the Speech-to-Text API V2 containing
        the transcription results.        
    """

    # Instantiates a client
    client = SpeechAsyncClient(
        client_options=ClientOptions(
            api_endpoint="us-central1-speech.googleapis.com",
        )
    )

    # Reads a file as bytes
    
    # with open(audio_file, "rb") as f:
    #     content = f.read()
    content = convert_to_wav(audio_file)

    # In practice, stream should be a generator yielding chunks of audio data
    chunk_length = len(content) // 1000
    stream = [
        content[start : start + chunk_length]
        for start in range(0, len(content), chunk_length)
    ]
    audio_requests = (
        cloud_speech.StreamingRecognizeRequest(audio=audio) for audio in stream
    )

    recognition_config = cloud_speech.RecognitionConfig(
        auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),        
        language_codes=["auto", ""],
        model="chirp_2",
    )
    streaming_config = cloud_speech.StreamingRecognitionConfig(
        streaming_features=cloud_speech.StreamingRecognitionFeatures(
            interim_results=True
        ),
        config=recognition_config
    )
    config_request = cloud_speech.StreamingRecognizeRequest(
        recognizer=f"projects/{PROJECT_ID}/locations/us-central1/recognizers/_",
        streaming_config=streaming_config,
    )

    def requests(config: cloud_speech.RecognitionConfig, audio: list) -> list:
        yield config
        yield from audio

    # Transcribes the audio into text
    responses_iterator = await client.streaming_recognize(
        requests=requests(config_request, audio_requests)
    )
    responses = []
    async for response in responses_iterator:
        responses.append(response)
        for result in response.results:
            print(f"Transcript: {result.alternatives[0].transcript}")
            print(f"Detected Language: {result.language_code}")
            print(f"Result: {result}")

    return responses

async def main():
    """Main async function to run the transcription."""
    await transcribe_streaming_chirp2(TEST_FILE_NAME)

if __name__ == "__main__":
    asyncio.run(main())