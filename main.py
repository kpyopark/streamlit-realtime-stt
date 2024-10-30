import streamlit as st
from pydub import AudioSegment
import threading
import queue
import asyncio
from google.cloud import speech_v2 as speech
from typing import Optional, Callable
import wave
import pyaudio
import time
import io
import os
from dotenv import load_dotenv
from google.api_core.client_options import ClientOptions

load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")
CHIRP_RECOGNIZER_ID = os.getenv("REGONIZER_ID")

class AudioProducer(threading.Thread):
    def __init__(self, wav_data: bytes, chunk_size: int = 1024):
        super().__init__()
        self.wav_data = wav_data
        self.chunk_size = chunk_size
        self.audio_queue = queue.Queue()
        self.is_playing = False
        self.daemon = True
        
    def run(self):
        """Thread's main method - handles audio streaming"""
        # wav_data를 BytesIO로 변환하여 메모리에서 직접 읽기
        wav_buffer = io.BytesIO(self.wav_data)
        with wave.open(wav_buffer, 'rb') as wf:
            p = pyaudio.PyAudio()
            
            # Open stream for playback
            stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                          channels=wf.getnchannels(),
                          rate=wf.getframerate(),
                          output=True)
            
            self.is_playing = True
            
            try:
                while self.is_playing:
                    data = wf.readframes(self.chunk_size)
                    if not data:
                        break
                        
                    # Play audio
                    stream.write(data)
                    
                    # Put audio data in queue for transcription
                    self.audio_queue.put(data)
                    
                    # Simulate real-time playback speed
                    #time.sleep(self.chunk_size / wf.getframerate())
                    
            finally:
                stream.stop_stream()
                stream.close()
                p.terminate()
                self.is_playing = False
                self.audio_queue.put(None)  # Sentinel value
    
    def stop(self):
        """Stop audio playback and streaming"""
        self.is_playing = False

class TranscriptionConsumer(threading.Thread):
    def __init__(self, 
                 audio_producer: AudioProducer,
                 language_code: str = "ko-KR",
                 on_transcription: Optional[Callable[[str, bool], None]] = None,
                 on_error: Optional[Callable[[Exception], None]] = None):
        super().__init__()
        self.audio_producer = audio_producer
        self.language_code = language_code
        self.on_transcription = on_transcription
        self.on_error = on_error
        self.is_running = False
        self.daemon = True
        self.loop = None

    async def process_audio_stream(self):
        """Process audio stream and handle transcription"""
        client = speech.SpeechAsyncClient(
          client_options=ClientOptions(
            api_endpoint=f"{LOCATION}-speech.googleapis.com",
          )
        )
        
        config = speech.types.RecognitionConfig(
            #encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            #sample_rate_hertz=16000,
            #enable_automatic_punctuation=True,
            auto_decoding_config=speech.types.AutoDetectDecodingConfig(),
            language_codes=["auto"], #[self.language_code],
            model="chirp_2"
        )
        
        streaming_config = speech.types.StreamingRecognitionConfig(
            config=config,
            streaming_features=speech.StreamingRecognitionFeatures(
                interim_results=True
            )
            # interim_results=True
        )

        config_request = speech.types.StreamingRecognizeRequest(
            recognizer=f"projects/{PROJECT_ID}/locations/{LOCATION}/recognizers/_",
            streaming_config=streaming_config
        )
        
        # 비동기 제너레이터 정의
        async def async_request_generator():
            # First request with configuration
            yield config_request
            
            # Process audio chunks from queue
            while self.is_running:
                try:
                    # 비동기적으로 큐에서 데이터 가져오기
                    chunk = await self.loop.run_in_executor(
                        None, 
                        self.audio_producer.audio_queue.get, 
                        True, 
                        1
                    )
                    #print("Receive Chunk.")
                    #print(chunk)
                    
                    if chunk is None:  # Check for sentinel value
                        print("###################")
                        print("###################")
                        print("Audio stream ended")
                        print("###################")
                        print("###################")
                        break
                    yield speech.types.StreamingRecognizeRequest(audio=chunk)
                except queue.Empty:
                    print("All Queue is Empty.")
                    continue
                except Exception as e:
                    print(e)
                    if self.on_error:
                        await self.loop.run_in_executor(None, self.on_error, e)
                    break
        try:
            print("###################")
            print("start_streaming_recongnition")
            # 비동기 스트리밍 인식 실행
            streaming_response = await client.streaming_recognize(
                requests=async_request_generator()
            )
            
            print("start_streaming_recongnition..2")
            # 비동기 스트리밍 인식 실행
            # 응답 처리
            async for response in streaming_response:
                print("###################")
                print("receive request")

                if not self.is_running:
                    break
                    
                for result in response.results:
                    if not result.alternatives:
                        continue
                        
                    transcript = result.alternatives[0].transcript
                    is_final = result.is_final
                    
                    if self.on_transcription:
                        # 콜백을 비동기적으로 실행
                        await self.loop.run_in_executor(
                            None,
                            self.on_transcription,
                            transcript,
                            is_final
                        )
                        
        except Exception as e:
            print(e)
            if self.on_error:
                await self.loop.run_in_executor(None, self.on_error, e)
        
        print("#####################")
        print("Ended Transcription.")
    
    def run(self):
        """Thread's main method - sets up asyncio event loop"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        self.is_running = True
        try:
            self.loop.run_until_complete(self.process_audio_stream())
        finally:
            print("#####################")
            print("Finally ------------------- exited.")
            self.is_running = False
            if self.loop and self.loop.is_running():
                # 실행 중인 모든 작업을 정리
                pending = asyncio.all_tasks(self.loop)
                self.loop.run_until_complete(asyncio.gather(*pending))
            if self.loop:
                self.loop.close()
    
    async def _stop_async(self):
        """비동기 정리 작업 수행"""
        self.is_running = False
        # 필요한 추가 비동기 정리 작업 수행
    
    def stop(self):
        """Stop transcription processing"""
        if self.loop and self.loop.is_running():
            self.loop.create_task(self._stop_async())
        self.is_running = False

class AudioTranscriptionManager:
    """Manager class to handle both Producer and Consumer threads"""
    def __init__(self, wav_data: bytes, language_code: str = "ko-KR"):
        self.producer = AudioProducer(wav_data)
        self.consumer = TranscriptionConsumer(
            audio_producer=self.producer,
            language_code=language_code,
            on_transcription=self._handle_transcription,
            on_error=self._handle_error
        )
        
    def _handle_transcription(self, transcript: str, is_final: bool):
        """Handle transcription results"""
        prefix = "Final: " if is_final else "Interim: "
        print(f"{prefix}{transcript}")
    
    def _handle_error(self, error: Exception):
        """Handle errors"""
        print(f"Error occurred: {str(error)}")
    
    def start(self):
        """Start both producer and consumer threads"""
        self.producer.start()
        self.consumer.start()
    
    def stop(self):
        """Stop both producer and consumer threads"""
        self.producer.stop()
        self.consumer.stop()
        
    def join(self):
        """Wait for both threads to complete"""
        self.producer.join()
        self.consumer.join()

# Streamlit 앱에서 사용할 때의 예시
def create_streamlit_transcription_manager(audio_file, language_code="ko-KR"):
    """Streamlit 앱용 트랜스크립션 매니저 생성 함수"""
    
    # 오디오 파일을 WAV로 변환
    audio = AudioSegment.from_file(audio_file)
    audio = audio.set_channels(1)  # mono
    audio = audio.set_frame_rate(8000)  # 16kHz
    
    # WAV 데이터를 메모리에 저장
    wav_buffer = io.BytesIO()
    audio.export(wav_buffer, format="wav")
    wav_data = wav_buffer.getvalue()
    
    # 상태 변수 초기화
    if 'full_transcript' not in st.session_state:
        st.session_state.full_transcript = ""
    
    def update_transcription(transcript: str, is_final: bool):
        if is_final:
            st.session_state.full_transcript += transcript + "\n"
            
        # Streamlit의 텍스트 영역 업데이트
        transcription_placeholder = st.session_state.get('transcription_placeholder')
        if transcription_placeholder:
            text = st.session_state.full_transcript
            if not is_final:
                text += f"*{transcript}*"
            transcription_placeholder.markdown(text)
    
    def handle_error(error: Exception):
        st.error(f"전사 중 오류가 발생했습니다: {str(error)}")
    
    # 트랜스크립션 매니저 생성
    manager = AudioTranscriptionManager(wav_data, language_code)
    manager.consumer.on_transcription = update_transcription
    manager.consumer.on_error = handle_error
    
    return manager

import streamlit as st
import time
from typing import Optional
from pydub import AudioSegment

def main():
    st.title("🎤 음성 전사 앱")
    
    # Initialize session state variables
    if 'transcription_manager' not in st.session_state:
        st.session_state.transcription_manager = None
    if 'is_transcribing' not in st.session_state:
        st.session_state.is_transcribing = False
    if 'full_transcript' not in st.session_state:
        st.session_state.full_transcript = ""
    
    # Callback functions for buttons
    def start_transcription():
        st.session_state.is_transcribing = True
        
    def stop_transcription():
        st.session_state.is_transcribing = False
        if st.session_state.transcription_manager:
            st.session_state.transcription_manager.stop()
            st.session_state.transcription_manager = None
    
    # Audio file upload
    audio_file = st.file_uploader("오디오 파일 선택 (MP3 또는 AAC)", type=['mp3', 'aac'])
    
    if audio_file:
        st.audio(audio_file)
        
        # Language selection
        language = st.selectbox(
            "언어 선택",
            options=["한국어", "영어", "중국어", "일본어"],
            format_func=lambda x: {
                "한국어": "한국어 (ko-KR)",
                "영어": "English (en-US)",
                "중국어": "简体中文 (zh-Hans-CN)",
                "일본어": "日本語 (ja-JP)"
            }[x]
        )
        
        language_codes = {
            "한국어": "ko-KR",
            "영어": "en-US",
            "중국어": "zh-Hans-CN",
            "일본어": "ja-JP"
        }
        
        # Control buttons in columns
        col1, col2 = st.columns(2)
        
        with col1:
            start_button = st.button(
                "변환 시작",
                disabled=st.session_state.is_transcribing,
                on_click=start_transcription,
                key='start_button'
            )
            
        with col2:
            stop_button = st.button(
                "변환 중지",
                disabled=not st.session_state.is_transcribing,
                on_click=stop_transcription,
                key='stop_button'
            )
        
        # Status indicator and transcription area
        status_container = st.container()
        transcription_container = st.container()
        
        with status_container:
            if st.session_state.is_transcribing:
                st.info("🔄 음성을 텍스트로 변환하는 중...")
            
        with transcription_container:
            st.markdown("### 변환 결과")
            transcription_placeholder = st.empty()
            st.session_state.transcription_placeholder = transcription_placeholder
            
            if st.session_state.full_transcript:
                transcription_placeholder.markdown(st.session_state.full_transcript)
        
        # Handle transcription process
        if st.session_state.is_transcribing and not st.session_state.transcription_manager:
            try:
                # Create and start transcription manager
                manager = create_streamlit_transcription_manager(
                    audio_file,
                    language_code=language_codes[language]
                )
                st.session_state.transcription_manager = manager
                manager.start()
                
            except Exception as e:
                st.error(f"오류가 발생했습니다: {str(e)}")
                stop_transcription()
        
        # Check completion
        if st.session_state.transcription_manager and not st.session_state.transcription_manager.producer.is_playing:
            stop_transcription()
            st.success("✅ 변환이 완료되었습니다.")
            
            if st.session_state.full_transcript:
                st.download_button(
                    label="텍스트 파일 다운로드",
                    data=st.session_state.full_transcript,
                    file_name=f"transcript_{language_codes[language]}.txt",
                    mime="text/plain"
                )
    
    # Sidebar information
    st.sidebar.markdown("""
    ### 💡 사용 방법
    1. AAC 또는 MP3 형식의 오디오 파일을 업로드합니다.
    2. 음성의 언어를 선택합니다.
    3. '변환 시작' 버튼을 클릭하면 실시간 변환이 시작됩니다.
    4. 언제든 '변환 중지' 버튼으로 중단할 수 있습니다.
    5. 변환이 완료되면 결과를 텍스트 파일로 다운로드할 수 있습니다.
    
    ### ⚠️ 주의사항
    - 파일 크기는 10MB 이하를 권장합니다.
    - 깨끗한 음성일수록 더 정확한 결과를 얻을 수 있습니다.
    - 모든 오디오는 16kHz 모노로 변환되어 처리됩니다.
    """
    )

if __name__ == "__main__":
    main()