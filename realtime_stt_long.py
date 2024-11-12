import streamlit as st
import os
import tempfile
from dataclasses import dataclass
from typing import List, Optional
import queue
from pathlib import Path
import time
from test_v2_long import TranscriptionService, convert_to_wav

@dataclass
class TranscriptionState:
    messages: List[dict]
    is_processing: bool
    error: Optional[str]
    temp_file_path: Optional[str]
    processed_file_path: Optional[str]
    message_queue: queue.Queue
    service: Optional[TranscriptionService]
    latest_transcript: str
    final_transcripts: List[str]
    is_completed: bool  # 전사 완료 상태를 추적하는 새 필드 추가

def initialize_session_state():
    if 'transcription_state' not in st.session_state:
        st.session_state.transcription_state = TranscriptionState(
            messages=[],
            is_processing=False,
            error=None,
            temp_file_path=None,
            processed_file_path=None,
            message_queue=queue.Queue(),
            service=None,
            latest_transcript="",
            final_transcripts=[],
            is_completed=False  # 초기값 설정
        )

def update_transcripts(state: TranscriptionState):
    """실시간 전사 내용 업데이트"""
    if not state.messages:
        return

    # 가장 최근 메시지의 전사 내용을 실시간으로 표시
    latest_msg = state.messages[-1]
    state.latest_transcript = latest_msg['transcript']

    # is_final이 'T'인 메시지들만 최종 전사 내용으로 저장
    state.final_transcripts = [
        msg['transcript'] 
        for msg in state.messages 
        if msg['is_final'] == 'T'
    ]
    
    # 전사 완료 여부 확인
    if state.service and not state.service.is_running:
        state.is_completed = True
        stop_transcription(state)
        st.success("전사가 완료되었습니다.")

def display_transcripts(state: TranscriptionState, realtime_container, final_container):
    """전사 내용 표시"""
    # 실시간 전사 내용 표시
    realtime_container.text_area(
        "실시간 전사 내용",
        state.latest_transcript,
        height=100
    )

    # 최종 전사 내용 표시
    if state.final_transcripts:
        final_text = "\n".join(state.final_transcripts)
        final_container.text_area(
            "최종 전사문",
            final_text,
            height=200
        )

def process_audio_file(file_path: str, denoising_methods: List[str]) -> str:
    """오디오 파일 전처리 함수"""
    try:
        processed_file_path = convert_to_wav(
            file_path, 
            sampling_rate=16000, 
            denoised_types=denoising_methods
        )
        if not processed_file_path:
            raise Exception("Failed to process audio file")
        return processed_file_path
    except Exception as e:
        raise Exception(f"Audio processing failed: {str(e)}")

def start_transcription(state: TranscriptionState):
    try:
        # TranscriptionService 초기화
        state.service = TranscriptionService(project_id=os.getenv("PROJECT_ID"))
        state.is_processing = True
        state.error = None
        state.messages = []
        state.latest_transcript = ""
        state.final_transcripts = []
        state.is_completed = False  # 전사 시작 시 완료 상태 초기화
        
        # 전사 시작
        state.service.start_transcription(
            audio_file=state.processed_file_path,
            location="us",
            model="long",
            language_code="cmn-Hans-CN",
            messages=state.messages
        )
        
    except Exception as e:
        state.error = str(e)
        state.is_processing = False
        if state.service:
            state.service.stop_transcription()

def stop_transcription(state: TranscriptionState):
    if state.service:
        state.service.stop_transcription()
        state.is_processing = False
        state.service = None

def cleanup_files(state: TranscriptionState):
    """파일 정리 함수"""
    if state.temp_file_path and os.path.exists(state.temp_file_path):
        os.unlink(state.temp_file_path)
        state.temp_file_path = None
    
    if state.processed_file_path and os.path.exists(state.processed_file_path):
        os.unlink(state.processed_file_path)
        state.processed_file_path = None

def main():
    initialize_session_state()
    state = st.session_state.transcription_state
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("음성 파일 업로드")
        uploaded_file = st.file_uploader(
            "음성 파일을 선택하세요", 
            type=['wav', 'mp3', 'aac'],
            key="file_uploader"
        )
        denoising_options = ['lowpass', 'equalized', 'smart', 'aggressive']
        selected_methods = st.multiselect(
            "디노이징 방법 선택",
            options=denoising_options,
            default=['lowpass'],
            help="여러 개의 디노이징 방법을 선택할 수 있습니다."
        )
        state.denoising_methods = selected_methods
        
        if uploaded_file and not state.is_processing:
            # 이전 파일들 정리
            cleanup_files(state)
            
            # 새 임시 파일 생성
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                state.temp_file_path = tmp_file.name
            
            try:
                # 파일 전처리
                with st.spinner("오디오 파일 전처리 중..."):
                    state.processed_file_path = process_audio_file(state.temp_file_path, state.denoising_methods)
                st.success("파일 전처리가 완료되었습니다.")
                
                # 전사 시작 버튼
                if st.button("전사 시작", key="transcribe_button"):
                    start_transcription(state)
                    
            except Exception as e:
                st.error(f"파일 처리 중 오류가 발생했습니다: {str(e)}")
                cleanup_files(state)
        
        # 전사 중지 버튼
        if state.is_processing and not state.is_completed:
            if st.button("전사 중지", key="realtime_stop_button"):
                stop_transcription(state)
                st.warning("전사가 중지되었습니다.")
    
    with col2:
        st.subheader("전사 결과")
        realtime_container = st.empty()
        final_container = st.empty()
        
        if state.is_processing and not state.is_completed:
            # 실시간 전사 내용 업데이트 및 표시
            update_transcripts(state)
            display_transcripts(state, realtime_container, final_container)
            st.info("전사 처리 중...")
            time.sleep(0.1)  # UI 업데이트를 위한 짧은 대기
            st.rerun()
        else:
            # 최종 전사 결과 표시
            display_transcripts(state, realtime_container, final_container)
        
        if state.error:
            st.error(f"전사 중 오류가 발생했습니다: {state.error}")

def cleanup_temp_files():
    state = st.session_state.get('transcription_state')
    if state:
        if state.service:
            state.service.stop_transcription()
        cleanup_files(state)

if __name__ == "__main__":
    st.set_page_config(page_title="음성 전사 시스템", layout="wide")
    st.title("음성 전사 시스템")
    
    try:
        main()
    finally:
        cleanup_temp_files()