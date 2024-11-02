import os
from dotenv import load_dotenv

from google.cloud.speech_v2 import SpeechClient, SpeechAsyncClient
from google.cloud.speech_v2.types import cloud_speech
from google.api_core.client_options import ClientOptions

from pydub import AudioSegment
import io

load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = 'us'
TEST_FILE_NAME = os.getenv("TEST_FILE_NAME")

SAMPLE_RATE = 16000

def convert_to_wav(audio_file):
    audio = AudioSegment.from_file(audio_file)
    audio = audio.set_channels(1)  # mono
    audio = audio.set_frame_rate(SAMPLE_RATE)  # 16kHz
    
    # WAV 데이터를 메모리에 저장
    wav_buffer = io.BytesIO()
    audio.export(wav_buffer, format="wav")
    wav_data = wav_buffer.getvalue()
    return wav_data

def transcribe_streaming_v2(
    audio_file: str,
    location: str,
    model: str,
    language_code: str
) -> cloud_speech.StreamingRecognizeResponse:
    # Instantiates a client
    client = SpeechAsyncClient(
        client_options=ClientOptions(
            api_endpoint=f"{location}-speech.googleapis.com",
        )
    )

    content = convert_to_wav(audio_file)

    chunk_length = SAMPLE_RATE # len(content) // 1000 # 25600 # SAMPLE_RATE # 
    stream = [
        content[start : start + chunk_length]
        for start in range(0, len(content), chunk_length)
    ]
    audio_requests = (
        cloud_speech.StreamingRecognizeRequest(audio=audio) for audio in stream
    )

    recognition_config = cloud_speech.RecognitionConfig(
        auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),        
        language_codes=[language_code],
        model=model,
    )
    streaming_config = cloud_speech.StreamingRecognitionConfig(
        streaming_features=cloud_speech.StreamingRecognitionFeatures(
            interim_results=True
        ),
        config=recognition_config
    )
    config_request = cloud_speech.StreamingRecognizeRequest(
        recognizer=f"projects/{PROJECT_ID}/locations/{location}/recognizers/_",
        streaming_config=streaming_config,
    )

    cnt = 0
    async def requests(config: cloud_speech.RecognitionConfig, audio: list) -> list:
        yield config
        for inx, audio_chunk in enumerate(audio):
            try:
                # await asyncio.sleep(0.99)
                print('feeding', inx)
                yield audio_chunk
            except Exception as e :
                e.with_traceback()

    # Transcribes the audio into text
    responses_iterator = await client.streaming_recognize(
        requests=requests(config_request, audio_requests)
    )
    responses = []
    print( "waiting response")
    async for response in responses_iterator:
        print("one response received.")
        responses.append(response)
        for result in response.results:
            # if 'alternatives' in result:
            #     print(f"Transcript: {result.alternatives[0].transcript}")
            # if 'language_code' in result:
            #     print(f"Detected Language: {result.language_code}")
            # if 'stability' in result:
            #     print(f'Stability: {result.stability}')
            print(f"Result: {result}")

    return responses

def main():
    transcribe_streaming_v2(TEST_FILE_NAME, "us", "long", "cmn-Hans-CN") ### OK
    ## await transcribe_streaming_v2(TEST_FILE_NAME, "us-central1", "chirp_2", "cmn-Hans-CN") ### Time Out 이후, google.api_core.exceptions.ServiceUnavailable: 503 502:Bad Gateway
    ## await transcribe_streaming_v2(TEST_FILE_NAME, "asia-southeast1", "chirp_2", "cmn-Hans-CN") ### X, 400 The model "chirp_2" does not exist in the location named "asia-southeast1". -- 문서상으로는 되는데...
    ## await transcribe_streaming_v2(TEST_FILE_NAME, "us-central1", "chirp_2", "cmn-Hans-CN") ### No error. But No Response ++ Added await asyncio.sleep


if __name__ == "__main__":
    asyncio.run(main())