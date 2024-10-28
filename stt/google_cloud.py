from .base import BaseSTTService, TranscriptionResult
from google.cloud import speech_v2 as speech
from google.api_core.client_options import ClientOptions
import os
from typing import AsyncGenerator
from dotenv import load_dotenv

load_dotenv()

class GoogleCloudSTTService(BaseSTTService):
    def __init__(self):
        self.project_id = os.getenv("PROJECT_ID")
        self.location = os.getenv("LOCATION")
        self.client = None
        self.isProgressing = False

    async def initialize(self):
        self.client = speech.SpeechAsyncClient(
            client_options=ClientOptions(
                api_endpoint=f"{self.location}-speech.googleapis.com",
            )
        )

    async def transcribe_stream(self, 
        audio_stream: AsyncGenerator[bytes, None],
        language_code: str
    ) -> AsyncGenerator[TranscriptionResult, None]:
        config = speech.types.RecognitionConfig(
            auto_decoding_config=speech.types.AutoDetectDecodingConfig(),
            language_codes=["auto"],
            model="chirp_2"
        )
        self.isProgressing = True
        
        streaming_config = speech.types.StreamingRecognitionConfig(
            config=config,
            streaming_features=speech.StreamingRecognitionFeatures(
                interim_results=True
            )
        )
        print(f"projects/{self.project_id}/locations/{self.location}/recognizers/_")
        config_request = speech.types.StreamingRecognizeRequest(
            recognizer=f"projects/{self.project_id}/locations/{self.location}/recognizers/_",
            streaming_config=streaming_config
        )

        async def request_generator():
            yield config_request
            async for chunk in audio_stream:
                #print('chunk is received.')
                if chunk:
                    yield speech.types.StreamingRecognizeRequest(audio=chunk)
                if not self.isProgressing:
                    break
        print('start recognition')
        streaming_response = await self.client.streaming_recognize(
            requests=request_generator()
        )
        print('waiting for response')
        async for response in streaming_response:
            print('receive transcription.')
            for result in response.results:
                print('receive result.')
                if result.alternatives:
                    print('receive result.', result.alternatives[0].transcript)
                    yield TranscriptionResult(
                        transcript=result.alternatives[0].transcript,
                        is_final=result.is_final,
                        confidence=result.alternatives[0].confidence,
                        language=language_code
                    )
        print('exit normally')

    async def cleanup(self):
        print('cleanup called')
        if self.client:
            self.isProgressing = False
            #await self.client.close()
