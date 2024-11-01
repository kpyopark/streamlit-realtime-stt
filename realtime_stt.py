import io
import streamlit as st
import time
import pandas as pd
from typing import Optional
from pydub import AudioSegment
import queue
from dataclasses import asdict

from stt.google_cloud import GoogleCloudSTTService
from stt.gemini_stt import GeminiSTTService
from stt.manager import AudioTranscriptionManager

from dotenv import load_dotenv

from llm_utility import GeminiAPI
import os
import json
import stt_utility as utility

from dotenv import load_dotenv

load_dotenv()


SAMPLING_RATE = int(os.getenv("SAMPLING_RATE"))
NCHANNEL = os.getenv("NCHANNEL")
CHUNK_DURATION_MS = 100
N_OVERWRAP_SEGMENT = 1
N_CHUNKS_IN_A_WINDOW = 41   # 4.1초


def main():
    #st.set_page_config(layout="wide")
    
    # Initialize session state variables
    if 'transcription_manager' not in st.session_state:
        st.session_state.transcription_manager = None
    if 'is_transcribing' not in st.session_state:
        st.session_state.is_transcribing = False
    if 'full_transcript' not in st.session_state:
        st.session_state.full_transcript = []
    if 'transcription_items' not in st.session_state:
        st.session_state.transcription_items = queue.Queue()
    if 'transcription_test_base' not in st.session_state:
        st.session_state.transcription_test_base = None
        
    # 제목 영역을 왼쪽 정렬
    left_title_col, right_title_col = st.columns([1, 3])
    with left_title_col:
        st.title("🎤 음성 전사 앱")


    def convert_to_wav():
        # 오디오 파일이 존재하는지 확인
        if 'audio_file' not in st.session_state:
            raise ValueError("오디오 파일이 업로드되지 않았습니다.")
        return utility.convert_to_wav(audio_file=st.session_state.audio_file, SAMPLING_RATE=SAMPLING_RATE)

    def create_streamlit_transcription_manager(audio_file, language_code="ko-KR"):
        """Streamlit 앱용 트랜스크립션 매니저 생성 함수"""
        
        audio = AudioSegment.from_file(audio_file)
        #audio = audio.set_channels(1)
        audio = audio.set_frame_rate(SAMPLING_RATE)
        
        wav_buffer = io.BytesIO()
        audio.export(wav_buffer, format="wav")
        wav_data = wav_buffer.getvalue()
        
        if 'full_transcript' not in st.session_state:
            st.session_state.full_transcript = []
        
        def handle_error(error: Exception):
            try:
                st.error(f"전사 중 오류가 발생했습니다: {str(error)}")
            except Exception as e:
                print(f"Error handling error: {e}")
        
        stt_service = GeminiSTTService()
        
        manager = AudioTranscriptionManager(
            wav_data=wav_data,
            stt_service=stt_service,
            language_code=language_code,
            chunk_duration_ms=CHUNK_DURATION_MS,
            overwrap_segment=N_OVERWRAP_SEGMENT,
            feeding_segment_window = N_CHUNKS_IN_A_WINDOW,
            need_wave_header = True,
            on_transcription=update_transcription,
            on_error=handle_error,
            message_queue=st.session_state.transcription_items
        )
        
        return manager

    def update_transcription(queue, transcript_item):
        queue.put(transcript_item)

    def start_transcription():
        st.session_state.is_transcribing = True
        st.session_state.full_transcript = []
        
    def stop_transcription():
        st.session_state.is_transcribing = False
        if st.session_state.transcription_manager:
            st.session_state.transcription_manager.stop()
            st.session_state.transcription_manager = None
    
    
    # Create two columns with custom width ratio
    left_col, right_col = st.columns([1, 3])
    
    with left_col:
        st.markdown("### ⚙️ 설정")

        transcription_text = st.text_area("Original Text (Ground Truth)", height=300)

        if st.button("테스트 자료 생성", type="primary"):
            if transcription_text:
                with st.spinner("테스트 데이터 생성 중..."):
                    # Gemini API 호출
                    gemini = GeminiAPI(PROJECT_ID, LOCATION)
                    transcription_test_base = gemini.transcription_to_testdata(transcription_text)
                    
                    if transcription_test_base:
                        # 메인 영역에 결과 표시
                        st.session_state.transcription_test_base = transcription_test_base
                    else:
                        st.error("테스트 데이터 생성에 실패했습니다.")
            else:
                st.warning("전사 원문을 입력해주세요.")

        # Audio file upload moved to the right column
        audio_file = st.file_uploader("오디오 파일 선택 (MP3 또는 AAC)", type=['mp3', 'aac'])

        if audio_file:
            st.session_state.audio_file = audio_file

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
        
        # Control buttons
        col1, col2, col3 = st.columns(3)
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
        with col3:
            convert_button = st.button(
                "WAV전환",
                disabled=st.session_state.is_transcribing,
                on_click=convert_to_wav,
                key='wav_convert'
            )

    with right_col:
            
        if st.session_state.is_transcribing:
            st.info("🔄 음성을 텍스트로 변환하는 중...")
        
        st.markdown("### 전사 결과")
        
        # 전사 결과를 DataFrame으로 변환하여 표시
        if st.session_state.full_transcript:
            df = pd.DataFrame([{
                'seq_id': item.seq_id if item.seq_id is not None else '',
                'timecode': item.timecode,
                'transcript': item.transcript,
                'confidence': item.confidence * 100,  # 백분율로 변환
                'is_final': item.is_final,
                'language': item.language,
                'translation': item.translation
            } for item in st.session_state.full_transcript])

            # DataFrame 컬럼 순서 조정
            df = df[[
                'seq_id', 
                'timecode', 
                'transcript', 
                'confidence', 
                'is_final', 
                'language', 
                'translation'
            ]]

            # 데이터프레임 표시
            st.dataframe(
                df,
                hide_index=True,
                use_container_width=True,
                height=500,
                column_config={
                    "seq_id": st.column_config.NumberColumn(
                        "순번",
                        help="음성 청크의 순서",
                        width="small"
                    ),
                    "timecode": st.column_config.TextColumn(
                        "시간",
                        help="음성 청크의 시작 시간",
                        width="small"
                    ),
                    "transcript": st.column_config.TextColumn(
                        "전사 내용",
                        help="음성을 텍스트로 변환한 내용",
                        width="large"
                    ),
                    "confidence": st.column_config.ProgressColumn(
                        "정확도",
                        help="전사 결과의 정확도",
                        format="%.0f%%",
                        min_value=0,
                        max_value=100,
                        width="small"
                    ),
                    "is_final": st.column_config.CheckboxColumn(
                        "최종 여부",
                        help="최종 전사 결과 여부",
                        width="small"
                    ),
                    "language": st.column_config.TextColumn(
                        "언어",
                        help="전사된 텍스트의 언어",
                        width="small"
                    ),
                    "translation": st.column_config.TextColumn(
                        "번역",
                        help="번역된 텍스트 (설정된 경우)",
                        width="large"
                    )
                }
            )

        if audio_file:
            if st.session_state.is_transcribing and not st.session_state.transcription_manager:
                try:
                    manager = create_streamlit_transcription_manager(
                        audio_file,
                        language_code=language_codes[language]
                    )
                    st.session_state.transcription_manager = manager
                    manager.start()
                    
                except Exception as e:
                    st.error(f"오류가 발생했습니다: {str(e)}")
                    stop_transcription()
            
            if st.session_state.transcription_manager and not st.session_state.transcription_manager.producer.is_playing:
                st.success("✅ 변환이 완료되었습니다.")
                
                if st.session_state.full_transcript:
                    # 최종 텍스트만 추출하여 다운로드용 텍스트 생성
                    full_text_content = "\n".join([
                        item.transcript 
                        for item in st.session_state.full_transcript 
                        if item.is_final
                    ])
                    # transcript 컬럼만 포함된 다운로드용 텍스트 생성
                    transcript_only_content = "\n".join([
                        item.transcript 
                        for item in st.session_state.full_transcript 
                        if item.is_final
                    ])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="전체 정보 다운로드",
                            data=full_text_content,
                            file_name=f"full_transcript_{language_codes[language]}.txt",
                            mime="text/plain"
                        )
                    with col2:
                        st.download_button(
                            label="Transcript만 다운로드",
                            data=transcript_only_content,
                            file_name=f"transcript_only_{language_codes[language]}.txt",
                            mime="text/plain"
                        )

        # 메인 영역에 결과 표시
        if 'transcription_test_base' in st.session_state:
            st.header("원문을 통해서 생성된 테스트 데이터")
            
            # 결과를 테이블 형식으로 표시
            if st.session_state.transcription_test_base:
                data_view = [{"순서": item["seq_id"], "텍스트": item["text"]} 
                            for item in st.session_state.transcription_test_base]
                st.table(data_view)
                
                # JSON 다운로드 버튼
                st.download_button(
                    label="JSON 파일 다운로드",
                    data=json.dumps(st.session_state.transcription_test_base, ensure_ascii=False, indent=2),
                    file_name="test_data.json",
                    mime="application/json"
                )

    # Queue processing logic
    if st.session_state.is_transcribing:
        time.sleep(0.1)
        try:
            while not st.session_state.transcription_items.empty():
                item = st.session_state.transcription_items.get_nowait()
                st.session_state.full_transcript.append(item)
                
        except queue.Empty:
            pass
        except Exception as e:
            st.error(f"Queue 처리 중 오류 발생: {str(e)}")
        st.rerun()
    


if __name__ == "__main__":
    main()