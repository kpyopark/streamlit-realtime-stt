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
                 chunk_duration_ms: int = 20, 
                 overwrap_segment: int = 0,
                 feeding_segment_window: int = 1,
                 need_wave_header:bool = False):  # Default 20ms chunks for real-time processing
        """
        Initialize AudioProducer with WAV data and chunk duration.
        
        Args:
            wav_data: WAV file data in bytes
            chunk_duration_ms: Duration of each chunk in milliseconds
        """
        super().__init__()
        self.wav_data = wav_data
        self.chunk_duration_ms = chunk_duration_ms
        self.overwrap_segment = overwrap_segment
        self.feeding_segment_window = feeding_segment_window
        self.need_wave_header = need_wave_header
        self.audio_queue: queue.Queue = queue.Queue(maxsize=100)  # Limit queue size to prevent memory issues
        self.is_playing = False
        self.daemon = True
        self._audio_config: Optional[AudioConfig] = None
        self._start_time: Optional[float] = None
        self._samples_played: int = 0
        self.audible_window_chunks = []         # audible window chunks. It will be used for feeding to STT service. The size of this list is feeding_segment_window.
        self.wf_param = None
        
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
        self.wf_param = wf.getparams()
        
    def _calculate_sleep_time(self) -> float:
        """Calculate how long to sleep to maintain real-time playback"""
        if not self._start_time:
            self._start_time = time.time()
            return 0
            
        elapsed_time = time.time() - self._start_time
        expected_time = self._samples_played / self._audio_config.sample_rate
        sleep_time = expected_time - elapsed_time
        
        return max(0, sleep_time)
    
    def make_audible_chunk_and_put(self, data: bytes):
        self.audible_window_chunks.append(data)
        if len(self.audible_window_chunks) >= self.feeding_segment_window:
            self.flush_audible_chunk()

    def calculate_nframes(self, bytes_chunk, channels, sampwidth):
        return bytes_chunk // (channels * sampwidth)   

    def flush_audible_chunk(self):
        with io.BytesIO() as wav_buffer:
            with wave.open(wav_buffer, 'wb') as output_buffer:
                #new_wf_param = self.wf_param
                #total_bytes = len(self.audible_window_chunks) * self._audio_config.chunk_size
                #new_wf_param.nframes = self.calculate_nframes(total_bytes, new_wf_param.nchannels, new_wf_param.sampwidth)
                output_buffer.setparams(self.wf_param)
                for data in self.audible_window_chunks:
                    output_buffer.writeframes(data)
                # Try to add to queue, skip if full
                try:
                    print("qsize:", self.audio_queue.qsize(), "buffer_size:", len(self.audible_window_chunks))
                    self.audio_queue.put(wav_buffer.getvalue(), block=False)
                    self.audible_window_chunks = self.audible_window_chunks[-self.overwrap_segment:]
                    print("buffer size:", len(self.audible_window_chunks), "overwrap_segment:", self.overwrap_segment)
                except queue.Full:
                    print("Warning: Audio queue is full, skipping chunk")
        
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
                    
                    self.make_audible_chunk_and_put(data)
                        
            finally:
                self.flush_audible_chunk()
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