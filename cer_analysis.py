import streamlit as st
import json
import pandas as pd
from llm_utility import GeminiAPI
from stt_utility import compute_cer, extract_transcript_from_googlestt_result

import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_ID = os.getenv('PROJECT_ID')
LOCATION = os.getenv('LOCATION')


def init_gemini_api():
    if 'gemini_api' not in st.session_state:
        st.session_state.gemini_api = GeminiAPI(
            project_id=PROJECT_ID,  # 실제 프로젝트 ID로 교체 필요
            location=LOCATION       # 실제 위치로 교체 필요
        )
    return st.session_state.gemini_api

def parse_json_safely(json_str):
    try:
        if isinstance(json_str, str):
            return json.loads(json_str)
        return json_str
    except json.JSONDecodeError as e:
        st.error(f"JSON 파싱 오류: {str(e)}")
        return None

def create_analysis_dataframe(transcription_statements):
    # 원문 분석 데이터를 DataFrame으로 변환
    df = pd.DataFrame(transcription_statements)
    
    # CER 계산
    def calculate_cer_with_stats(row):
        print('rowvalue:',row)
        if pd.notna(row['transcription']):
            cer, stats = compute_cer(row['original_text'], row['transcription'])
            return pd.Series({
                'cer': f"{cer*100:.2f}%",
                'substitutions': stats['Substitutions'],
                'deletions': stats['Deletions'],
                'insertions': stats['Insertions']
            })
        return pd.Series({
            'cer': None,
            'substitutions': None,
            'deletions': None,
            'insertions': None
        })
    
    # CER 및 상세 통계 계산
    error_stats = df.apply(calculate_cer_with_stats, axis=1)
    df = pd.concat([df, error_stats], axis=1)
    
    return df

def show():
    st.header("음성 전사 CER 분석")
    
    # Initialize Gemini API
    gemini_api = init_gemini_api()
    
    # 좌우 컬럼 생성
    left_col, right_col = st.columns([1, 1])
    
    with left_col:
        st.subheader("입력")
        # 원문 입력
        original_text = st.text_area(
            "원문 (Original Text)",
            height=200,
            help="분석할 원문을 입력하세요."
        )
        
        # 전사 결과 JSON 입력
        transcription_json = st.text_area(
            "전사 결과 (JSON)",
            height=200,
            help="Google STT 결과 JSON을 입력하세요."
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 원문 분석 버튼
            if st.button("원문 분석", key="analyze_original"):
                if original_text:
                    with st.spinner("원문을 분석 중입니다..."):
                        # 원문 문장 분리 수행
                        result = gemini_api.split_original_text_to_statements(original_text)
                        if result:
                            st.session_state.original_statements = result
                            st.success("원문 분석이 완료되었습니다.")
                else:
                    st.warning("원문을 입력해주세요.")
        
        with col2:
            # 전사 분석 버튼
            if st.button("전사 분석", key="analyze_transcription"):
                if transcription_json and 'original_statements' in st.session_state:
                    with st.spinner("전사 결과를 분석 중입니다..."):
                        # 전사 결과 JSON 파싱
                        trans_data = parse_json_safely(transcription_json)
                        if trans_data:
                            # JSON에서 전사 텍스트 추출
                            transcript_text = extract_transcript_from_googlestt_result(trans_data)
                            if transcript_text:
                                # 추출된 텍스트를 원문과 동일한 방식으로 문장 분리
                                result = gemini_api.transcription_to_testdata(st.session_state.original_statements, transcript_text)
                                if result and 'final' in result:
                                    st.session_state.transcription_statements = result['final']
                                    st.success("전사 분석이 완료되었습니다.")
                else:
                    if 'original_statements' not in st.session_state:
                        st.warning("먼저 원문 분석을 수행해주세요.")
                    else:
                        st.warning("전사 결과를 입력해주세요.")
    
    with right_col:
        st.subheader("분석 결과")
        
        # 분석 결과 표시
        if 'original_statements' in st.session_state and 'transcription_statements' in st.session_state:
            # DataFrame 생성 및 표시
            df = create_analysis_dataframe(
                st.session_state.transcription_statements
            )
            
            print(df)
            # 기본 결과 표시
            st.dataframe(
                df.style.highlight_max(axis=0, subset=['cer'], color='red')
                    .highlight_min(axis=0, subset=['cer'], color='green'),
                hide_index=True,
                use_container_width=True
            )
            
            # 전체 통계 표시
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_cer = df['cer'].str.rstrip('%').astype(float).mean()
                st.metric("평균 CER", f"{avg_cer:.2f}%")
            
            with col2:
                total_subs = df['substitutions'].sum()
                st.metric("총 치환 수", f"{total_subs:,}")
            
            with col3:
                total_dels = df['deletions'].sum()
                st.metric("총 삭제 수", f"{total_dels:,}")
            
            with col4:
                total_ins = df['insertions'].sum()
                st.metric("총 삽입 수", f"{total_ins:,}")
            
            # 결과 다운로드 버튼
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="분석 결과 다운로드",
                data=csv,
                file_name="cer_analysis_result.csv",
                mime="text/csv",
                help="분석 결과를 CSV 파일로 다운로드합니다."
            )
        else:
            st.info("원문과 전사 분석을 모두 완료하면 결과가 여기에 표시됩니다.")