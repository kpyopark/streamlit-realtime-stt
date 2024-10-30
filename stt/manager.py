from .audio_producer import AudioProducer
from .transcription_consumer import TranscriptionConsumer
from .base import BaseSTTService
from typing import Optional, Callable
from .audio_config import AudioConfig
import queue

class AudioTranscriptionManager:
    def __init__(self, 
                 wav_data: bytes, 
                 stt_service: BaseSTTService,
                 language_code: str = "ko-KR",
                 chunk_duration_ms: int = 100,
                 overwrap_segment: int = 2,
                 feeding_segment_window: int = 20,
                 need_wave_header:bool = True,
                 on_transcription: Optional[Callable[[str, bool], None]] = None,
                 on_error: Optional[Callable[[Exception], None]] = None,
                 message_queue: queue.Queue = None
                 ):
        self.producer = AudioProducer(
            wav_data, 
            chunk_duration_ms=chunk_duration_ms,
            overwrap_segment=overwrap_segment,
            feeding_segment_window=feeding_segment_window,
            need_wave_header=need_wave_header
        )
        self.consumer = TranscriptionConsumer(
            audio_producer=self.producer,
            stt_service=stt_service,
            language_code=language_code,
            on_transcription=on_transcription or self._handle_transcription,
            on_error=on_error or self._handle_error,
            message_queue=message_queue
        )
        
    @property
    def audio_config(self) -> Optional[AudioConfig]:
        return self.producer.audio_config
        
    def _handle_transcription(self, transcript: str, is_final: bool):
        prefix = "Final: " if is_final else "Interim: "
        print(f"{prefix}{transcript}")
    
    def _handle_error(self, error: Exception):
        print(f"Error occurred: {str(error)}")
    
    def start(self):
        self.producer.start()
        self.consumer.start()
    
    def stop(self):
        self.producer.stop()
        self.consumer.stop()
        
    def join(self):
        self.producer.join()
        self.consumer.join()
