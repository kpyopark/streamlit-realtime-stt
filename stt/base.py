from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional, Callable
from dataclasses import dataclass

@dataclass
class TranscriptionResult:
    transcript: str
    is_final: str
    confidence: float = 0.0
    language: str = ""
    seq_id: Optional[int] = None
    timecode: str = ""
    translation: str = ""


class BaseSTTService(ABC):
    """Base class for all STT services"""
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize any necessary clients or resources"""
        pass

    @abstractmethod
    async def transcribe_stream(self, 
        audio_stream: AsyncGenerator[bytes, None],
        language_code: str
    ) -> AsyncGenerator[TranscriptionResult, None]:
        """Transcribe streaming audio data"""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup any resources"""
        pass