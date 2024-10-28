import threading
import queue
import wave
import pyaudio
import io
import time
from typing import Optional
from .audio_config import AudioConfig

class AudioProducer(threading.Thread):
    def __init__(self, 
                 wav_data: bytes,
                 chunk_duration_ms: int = 20):  # Default 20ms chunks for real-time processing
        """
        Initialize AudioProducer with WAV data and chunk duration.
        
        Args:
            wav_data: WAV file data in bytes
            chunk_duration_ms: Duration of each chunk in milliseconds
        """
        super().__init__()
        self.wav_data = wav_data
        self.chunk_duration_ms = chunk_duration_ms
        self.audio_queue: queue.Queue = queue.Queue(maxsize=100)  # Limit queue size to prevent memory issues
        self.is_playing = False
        self.daemon = True
        self._audio_config: Optional[AudioConfig] = None
        self._start_time: Optional[float] = None
        self._samples_played: int = 0
        
    @property
    def audio_config(self) -> Optional[AudioConfig]:
        return self._audio_config
        
    def _init_audio_config(self, wf: wave.Wave_read) -> None:
        """Initialize audio configuration from wave file"""
        self._audio_config = AudioConfig(
            channels=wf.getnchannels(),
            sample_rate=wf.getframerate(),
            sample_width=wf.getsampwidth(),
            chunk_duration_ms=self.chunk_duration_ms
        )
        
    def _calculate_sleep_time(self) -> float:
        """Calculate how long to sleep to maintain real-time playback"""
        if not self._start_time:
            self._start_time = time.time()
            return 0
            
        elapsed_time = time.time() - self._start_time
        expected_time = self._samples_played / self._audio_config.sample_rate
        sleep_time = expected_time - elapsed_time
        
        return max(0, sleep_time)
        
    def run(self):
        """Thread's main method - handles audio streaming"""
        wav_buffer = io.BytesIO(self.wav_data)
        
        with wave.open(wav_buffer, 'rb') as wf:
            self._init_audio_config(wf)
            chunk_size = self._audio_config.chunk_size
            
            p = pyaudio.PyAudio()
            stream = p.open(
                format=p.get_format_from_width(self._audio_config.sample_width),
                channels=self._audio_config.channels,
                rate=self._audio_config.sample_rate,
                output=True,
                frames_per_buffer=chunk_size
            )
            
            self.is_playing = True
            self._start_time = None
            self._samples_played = 0
            
            is_first_frame = True

            try:
                while self.is_playing:
                    data = wf.readframes(chunk_size)
                    if not data:
                        break
                    
                    # Calculate and sleep for real-time simulation
                    sleep_time = self._calculate_sleep_time()
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    
                    # Play audio
                    stream.write(data)
                    
                    # Update tracking variables
                    self._samples_played += chunk_size
                    
                    # Try to add to queue, skip if full
                    try:
                        if is_first_frame:
                            #is_first_frame = False
                            with io.BytesIO() as wav_buffer:
                                with wave.open(wav_buffer, 'wb') as output_buffer:
                                    output_buffer.setparams(wf.getparams())
                                    output_buffer.writeframes(data)
                                self.audio_queue.put(wav_buffer.getvalue(), block=False)
                        else:
                            self.audio_queue.put(data, block=False)
                    except queue.Full:
                        print("Warning: Audio queue is full, skipping chunk")
                        continue
                        
            finally:
                stream.stop_stream()
                stream.close()
                p.terminate()
                self.is_playing = False
                # Signal end of stream
                try:
                    self.audio_queue.put(None, block=False)
                except queue.Full:
                    pass
    
    def stop(self):
        """Stop audio playback and streaming"""
        self.is_playing = False
        
    def get_audio_config(self) -> Optional[AudioConfig]:
        """Get audio configuration if initialized"""
        return self._audio_config