# stt/audio_config.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class AudioConfig:
    """Audio configuration parameters"""
    channels: int
    sample_rate: int
    sample_width: int
    chunk_duration_ms: int = 20  # Default 20ms chunks
    
    @property
    def chunk_size(self) -> int:
        """Calculate chunk size based on sample rate and desired chunk duration"""
        return int((self.sample_rate * self.chunk_duration_ms) / 1000)
    
    @property
    def bytes_per_sample(self) -> int:
        """Calculate bytes per sample"""
        return self.sample_width * self.channels