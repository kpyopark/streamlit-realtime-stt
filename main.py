import streamlit as st
import realtime_stt as realtime_transcription
import cer_analysis
import realtime_stt_long as long_realtime


import os
import json
from dotenv import load_dotenv

load_dotenv()

PROJECT_ID = os.getenv('PROJECT_ID')
LOCATION = os.getenv('LOCATION')

def main():
    st.set_page_config(
        page_title="음성 전사 분석 시스템",
        page_icon="🎤",
        layout="wide"
    )
    
    # 앱 제목
    st.title("음성 전사 분석 시스템")
    
    # 탭 생성
    tab1, tab2, tab3 = st.tabs(["실시간 전사", "CER 분석", "LONG Realtime"])
    
    # 각 탭에 해당하는 페이지 콘텐츠 로드
    with tab1:
        realtime_transcription.main()
    
    with tab2:
        cer_analysis.show()

    with tab3:
        long_realtime.main()

if __name__ == "__main__":
    main()
