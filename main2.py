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

def main():
    st.set_page_config(layout="wide")
    
    # Initialize session state variables
    if 'transcription_manager' not in st.session_state:
        st.session_state.transcription_manager = None
    if 'is_transcribing' not in st.session_state:
        st.session_state.is_transcribing = False
    if 'full_transcript' not in st.session_state:
        st.session_state.full_transcript = []
    if 'transcription_items' not in st.session_state:
        st.session_state.transcription_items = queue.Queue()
        
    # 제목 영역을 왼쪽 정렬
    left_title_col, right_title_col = st.columns([1, 3])
    with left_title_col:
        st.title("🎤 음성 전사 앱")

    def create_streamlit_transcription_manager(audio_file, language_code="ko-KR"):
        """Streamlit 앱용 트랜스크립션 매니저 생성 함수"""
        
        audio = AudioSegment.from_file(audio_file)
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(16000)
        
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
            chunk_duration_ms=2000,
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
        # Audio file upload and controls
        audio_file = st.file_uploader("오디오 파일 선택 (MP3 또는 AAC)", type=['mp3', 'aac'])
        
        if audio_file:
            st.audio(audio_file)
            
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
                    text_content = "\n".join([
                        item.transcript 
                        for item in st.session_state.full_transcript 
                        if item.is_final
                    ])
                    st.download_button(
                        label="텍스트 파일 다운로드",
                        data=text_content,
                        file_name=f"transcript_{language_codes[language]}.txt",
                        mime="text/plain"
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