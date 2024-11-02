import os
from dotenv import load_dotenv

from google.cloud.speech_v2 import SpeechClient, SpeechAsyncClient
from google.cloud.speech_v2.types import cloud_speech
from google.api_core.client_options import ClientOptions

from pydub import AudioSegment
import io
import asyncio
import time
from queue import Queue
import threading

load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = 'us'
TEST_FILE_NAME = os.getenv("TEST_FILE_NAME")

SAMPLE_RATE = 16000

# def transcribe_sync_chirp2_auto_detect_language(
#     audio_file: str
# ) -> cloud_speech.RecognizeResponse:
#     """Transcribes an audio file and auto-detect spoken language using Chirp 2.
#     Please see https://cloud.google.com/speech-to-text/v2/docs/encoding for more
#     information on which audio encodings are supported.
#     Args:
#         audio_file (str): Path to the local audio file to be transcribed.
#             Example: "resources/audio.wav"
#     Returns:
#         cloud_speech.RecognizeResponse: The response from the Speech-to-Text API containing
#         the transcription results.
#     """
#     # Instantiates a client
#     client = SpeechClient(
#         client_options=ClientOptions(
#             api_endpoint=f"{LOCATION}-speech.googleapis.com",
#         )
#     )

#     RECOGNIZER = client.recognizer_path(PROJECT_ID, LOCATION, "_")
#     # Reads a file as bytes
#     with open(audio_file, "rb") as f:
#         audio_content = f.read()

#     config = cloud_speech.RecognitionConfig(
#         auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
#         language_codes=["cmn-Hans-CN"], # ['zh-TW', 'zh-CN'],  # Set language code to auto to detect language.
#         model="chirp",
#     )
#     print(RECOGNIZER)
#     request = cloud_speech.RecognizeRequest(
#         recognizer=RECOGNIZER,
#         config=config,
#         content=audio_content,
#     )

#     # Transcribes the audio into text
#     response = client.recognize(request=request)

#     for result in response.results:
#         print(f"Transcript: {result.alternatives[0].transcript}")
#         print(f"Detected Language: {result.language_code}")

#     return response

def convert_to_wav(audio_file):
    audio = AudioSegment.from_file(audio_file)
    audio = audio.set_channels(1)  # mono
    audio = audio.set_frame_rate(SAMPLE_RATE)  # 16kHz
    
    # WAV 데이터를 메모리에 저장
    wav_buffer = io.BytesIO()
    audio.export(wav_buffer, format="wav")
    wav_data = wav_buffer.getvalue()
    return wav_data

async def transcribe_streaming_v2(
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

    chunk_length = SAMPLE_RATE//10 # len(content) // 1000 # 25600 # SAMPLE_RATE # 100ms
    chunk_duration_ms = 1000 // (SAMPLE_RATE // chunk_length)

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

    start_time = time.time()

    async def calculate_waittime(inx, previous_time):
        actual_time = (inx * chunk_duration_ms) / 1000
        elapsed_time = time.time() - start_time
        waiting_time = actual_time - elapsed_time
        print(waiting_time)
        return max(0,waiting_time)

    async def requests(config: cloud_speech.RecognitionConfig, audio: list) -> list:
        yield config
        for inx, audio_chunk in enumerate(audio):
            try:
                actual_time = (inx * chunk_duration_ms) / 1000
                elapsed_time = time.time() - start_time
                waiting_time = actual_time - elapsed_time
                await asyncio.sleep(waiting_time)
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

def transcribe_streaming_v2_sync(
    audio_file: str,
    location: str,
    model: str,
    language_code: str
) -> cloud_speech.StreamingRecognizeResponse:
    # Instantiates a client
    client = SpeechClient(
        client_options=ClientOptions(
            api_endpoint=f"{location}-speech.googleapis.com",
        )
    )

    content = convert_to_wav(audio_file)

    chunk_length = SAMPLE_RATE//10
    chunk_duration_ms = 1000 // (SAMPLE_RATE // chunk_length)

    # 오디오 청크를 저장할 큐 생성
    audio_queue = Queue()
    
    # 피딩 완료를 나타내는 이벤트
    feeding_done = threading.Event()

    def audio_feeder():
        start_time = time.time()
        cnt = 0        
        # 오디오 데이터를 청크로 나누어 큐에 넣기
        for start in range(0, len(content), chunk_length):
            chunk = content[start:start + chunk_length]
            
            # 실제 시간과 경과 시간을 계산하여 적절한 딜레이 추가
            actual_time = (start // chunk_length * chunk_duration_ms) / 1000
            elapsed_time = time.time() - start_time
            wait_time = max(0, actual_time - elapsed_time)
            
            if wait_time > 0:
                time.sleep(wait_time)
            
            audio_request = cloud_speech.StreamingRecognizeRequest(audio=chunk)
            audio_queue.put(audio_request)
            cnt = cnt + 1
            if cnt % 10 == 0:
                print(f'feeding chunk {start // chunk_length}')
        
        # 피딩 완료 표시
        feeding_done.set()
        print("Audio feeding completed")

    def request_generator(config: cloud_speech.StreamingRecognizeRequest) -> list:
        # 설정 요청 먼저 전송
        yield config
        
        # 큐에서 오디오 청크를 가져와서 전송
        while not (feeding_done.is_set() and audio_queue.empty()):
            try:
                audio_chunk = audio_queue.get(timeout=1.0)
                yield audio_chunk
                #print('chunk sent to recognition')
            except Queue.Empty:
                continue

    # Recognition 설정
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

    # 오디오 피딩 쓰레드 시작
    feeder_thread = threading.Thread(target=audio_feeder)
    feeder_thread.start()

    # Transcription 실행
    responses = []
    print("Starting recognition...")
    
    try:
        responses_iterator = client.streaming_recognize(
            requests=request_generator(config_request)
        )
        
        for response in responses_iterator:
            print("Response received.")
            responses.append(response)
            for result in response.results:
                print(f"Result: {result}")
    
    finally:
        # 쓰레드 종료 대기
        feeder_thread.join()

    return responses

async def main():
    """Main async function to run the transcription."""
    # await transcribe_streaming_v2(TEST_FILE_NAME, "us", "long", "cmn-Hans-CN") ### OK
    ## await transcribe_streaming_v2(TEST_FILE_NAME, "us-central1", "chirp_2", "cmn-Hans-CN") ### Time Out 이후, google.api_core.exceptions.ServiceUnavailable: 503 502:Bad Gateway
    ## await transcribe_streaming_v2(TEST_FILE_NAME, "asia-southeast1", "chirp_2", "cmn-Hans-CN") ### X, 400 The model "chirp_2" does not exist in the location named "asia-southeast1". -- 문서상으로는 되는데...
    ## await transcribe_streaming_v2(TEST_FILE_NAME, "us-central1", "chirp_2", "cmn-Hans-CN") ### No error. But No Response ++ Added await asyncio.sleep
    await transcribe_streaming_v2_sync(TEST_FILE_NAME, "us", "long", "cmn-Hans-CN") ### OK

def sync_main():
    _thread = threading.Thread(target=transcribe_streaming_v2_sync, args=[TEST_FILE_NAME, "us", "long", "cmn-Hans-CN"])
    _thread.start()    

    # transcribe_streaming_v2_sync(TEST_FILE_NAME, "us", "long", "cmn-Hans-CN") ### OK

if __name__ == "__main__":
    #asyncio.run(main())
    sync_main()
