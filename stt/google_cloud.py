from .base import BaseSTTService, TranscriptionResult
from google.cloud import speech_v2 as speech
from google.api_core.client_options import ClientOptions
from queue import Queue
import os
import threading
import asyncio
from typing import AsyncGenerator, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

class GoogleCloudSTTService(BaseSTTService):
    def __init__(self):
        self.project_id = os.getenv("PROJECT_ID")
        self.location = 'us'
        self.model = 'long'
        self.client = None
        self.isProgressing = False
        self.audio_queue: Queue[Optional[bytes]] = Queue()
        self.result_queue: Queue[Optional[Dict]] = Queue()
        self.should_stop = False
        self._thread: Optional[threading.Thread] = None
            
    async def initialize(self):
        pass

    def _recognition_thread(self):
        self.client = speech.SpeechClient(
            client_options=ClientOptions(
                api_endpoint=f"{self.location}-speech.googleapis.com",
            )
        )
        self.config = speech.types.RecognitionConfig(
            auto_decoding_config=speech.types.AutoDetectDecodingConfig(),
            language_codes=['cmn-Hans-CN'],
            model=self.model
        )
        self.streaming_config = speech.types.StreamingRecognitionConfig(
            config=self.config,
            streaming_features=speech.StreamingRecognitionFeatures(
                interim_results=True
            )
        )
        self.config_request = speech.types.StreamingRecognizeRequest(
            recognizer=f"projects/{self.project_id}/locations/{self.location}/recognizers/_",
            streaming_config=self.streaming_config,
        )

        def request_generator():
            yield self.config_request

            while True:
                try:
                    chunk = self.audio_queue.get(timeout=1.0)
                    if chunk is None:  # Sentinel value to stop the generator
                        break
                except queue.Empty:
                    continue
                print('feeding')
                yield speech.types.StreamingRecognizeRequest(audio=chunk)
                print('feeded')

        try:
            print('calling streaming request synchronously.')
            responses = self.client.streaming_recognize(requests=request_generator())
            print('waiting response.')
            for response in responses:
                print('one response is received.')
                for result in response.results:
                    print('result:', result)
                    if result.alternatives:
                        self.result_queue.put({
                            'transcript': result.alternatives[0].transcript,
                            'is_final': result.is_final,
                            'confidence': result.alternatives[0].confidence
                        })
            print('exit response thread.')
        except Exception as e:
            e.with_traceback()
            print(f"Error in recognition thread: {e}")
        finally:
            self.result_queue.put(None)  # Sentinel value to indicate completion

    async def transcribe_stream(self, 
        audio_stream: AsyncGenerator[bytes, None],
        language_code: str
    ) -> AsyncGenerator[TranscriptionResult, None]:
        # Start recognition thread
        self.isProgressing = True
        self._thread = threading.Thread(target=self._recognition_thread)
        self._thread.start()

        # Start audio processing task
        async def process_audio():
            try:
                async for chunk in audio_stream:
                    if not self.isProgressing:
                        break
                    self.audio_queue.put(chunk)
            finally:
                self.audio_queue.put(None)  # Signal end of audio stream

        # Start result processing task
        async def process_results():
            while True:
                # Use run_in_executor to get results from queue without blocking
                result = await asyncio.get_event_loop().run_in_executor(
                    None, self.result_queue.get)
                if result is None:  # Check for sentinel value
                    break
                yield TranscriptionResult(
                    transcript=result['transcript'],
                    is_final=result['is_final'],
                    confidence=result['confidence'],
                    language=language_code
                )

        # Run both tasks concurrently
        audio_task = asyncio.create_task(process_audio())
        async for result in process_results():
            yield result
        
        # Wait for audio processing to complete
        await audio_task

    async def cleanup(self):
        self.isProgressing = False
        if self._thread and self._thread.is_alive():
            self.audio_queue.put(None)  # Signal thread to stop
            self._thread.join()
            self._thread = None