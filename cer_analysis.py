import streamlit as st
import json
import pandas as pd
from llm_utility import GeminiAPI
from stt_utility import compute_cer, extract_transcript_from_googlestt_result, clean_chinese_text

import os
import io
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

def parse_input_data(file_type, input_text):
    """
    입력 데이터를 파싱하여 transcript 텍스트를 추출합니다.
    JSON 또는 CSV 형식을 지원합니다.
    """
    # CSV 형식인지 먼저 확인
    if file_type == 'csv' :
        try:
            # CSV 스트링을 DataFrame으로 변환 시도
            df = pd.read_csv(io.StringIO(input_text))
            if 'transcript' in df.columns:
                # transcript 컬럼의 모든 텍스트를 연결
                return ' '.join(df['transcript'].dropna().astype(str))
            else:
                st.error("CSV 파일에 'transcript' 컬럼이 없습니다.")
                return None
        except Exception as csve:
            print("CSV 파일 처리중 오류")
            print(csve)
            return None
    else:
        # CSV 파싱 실패 시 JSON 파싱 시도
        try:
            if isinstance(input_text, str):
                json_data = json.loads(input_text)
                return extract_transcript_from_googlestt_result(json_data)
            return extract_transcript_from_googlestt_result(input_text)
        except json.JSONDecodeError as e:
            print(e)
            st.error(f"입력 데이터 파싱 오류: 올바른 CSV 또는 JSON 형식이 아닙니다. {str(e)}")
            return None

def create_analysis_dataframe(transcription_statements):
    # 원문 분석 데이터를 DataFrame으로 변환
    df = pd.DataFrame(transcription_statements)
    
    # 클렌징된 텍스트 컬럼 추가
    df['cleaned_original'] = df['original_text'].apply(clean_chinese_text)
    df['cleaned_transcription'] = df['transcription'].apply(lambda x: clean_chinese_text(x) if pd.notna(x) else x)
    
    # CER 및 문장 일치 여부 계산
    def calculate_metrics_with_stats(row):
        if pd.notna(row['transcription']):
            cer, stats = compute_cer(row['original_text'], row['transcription'])
            # 문장 단위 오류 여부 (완벽히 일치하면 0, 아니면 1)
            is_sentence_error = 1 if cer > 0 else 0
            return pd.Series({
                'cer': f"{cer*100:.2f}%",
                'substitutions': stats['Substitutions'],
                'deletions': stats['Deletions'],
                'insertions': stats['Insertions'],
                'is_sentence_error': is_sentence_error
            })
        return pd.Series({
            'cer': None,
            'substitutions': None,
            'deletions': None,
            'insertions': None,
            'is_sentence_error': None
        })
    
    # CER 및 상세 통계 계산
    metrics_stats = df.apply(calculate_metrics_with_stats, axis=1)
    df = pd.concat([df, metrics_stats], axis=1)
    
    return df

def show():
    st.header("음성 전사 CER/SER 분석")
    
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

        result_type = st.selectbox(
            "전사 결과 File Type",
            options=["json", "csv"],
            format_func=lambda x: {
                "json": "json",
                "csv": "csv (transcript column 사용)"
            }[x]
        )
        
        # 전사 결과 JSON 입력
        transcription_input = st.text_area(
            "전사 결과 (JSON 또는 CSV)",
            height=200,
            help="Google STT 결과 JSON 또는 CSV 형식의 데이터를 입력하세요."
        )

        col1, col2 = st.columns(2)
        
        with col1:
            # 원문 분석 버튼
            if st.button("원문 분석", key="analyze_original"):
                if original_text:
                    with st.spinner("원문을 분석 중입니다..."):
                        result = gemini_api.split_original_text_to_statements(original_text)
                        if result:
                            st.session_state.original_statements = result
                            st.success("원문 분석이 완료되었습니다.")
                else:
                    st.warning("원문을 입력해주세요.")
        
        with col2:
            # 전사 분석 버튼
            if st.button("전사 분석", key="analyze_transcription"):
                if transcription_input and 'original_statements' in st.session_state:
                    with st.spinner("전사 결과를 분석 중입니다..."):
                        transcript_text = parse_input_data(result_type, transcription_input)
                        if transcript_text:
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
        
        if 'original_statements' in st.session_state and 'transcription_statements' in st.session_state:
            # DataFrame 생성 및 표시
            df = create_analysis_dataframe(
                st.session_state.transcription_statements
            )
            
            # 탭 생성
            tab1, tab2 = st.tabs(["기본 정보", "클렌징된 텍스트"])
            
            with tab1:
                # 기본 정보 표시
                base_columns = ['original_text', 'transcription', 'cer', 'substitutions', 'deletions', 'insertions']
                st.dataframe(
                    df[base_columns].style.highlight_max(axis=0, subset=['cer'], color='red')
                        .highlight_min(axis=0, subset=['cer'], color='green'),
                    hide_index=True,
                    use_container_width=True
                )
            
            with tab2:
                # 클렌징된 텍스트 정보 표시
                cleaning_columns = ['original_text', 'cleaned_original', 'transcription', 'cleaned_transcription']
                st.dataframe(
                    df[cleaning_columns],
                    hide_index=True,
                    use_container_width=True
                )
            
            # 전체 통계 표시
            st.subheader("통계 요약")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                avg_cer = df['cer'].str.rstrip('%').astype(float).mean()
                st.metric("평균 CER", f"{avg_cer:.2f}%")
            
            with col2:
                # SER 계산 (에러가 있는 문장 수 / 전체 문장 수)
                ser = (df['is_sentence_error'].sum() / len(df)) * 100
                st.metric("SER", f"{ser:.2f}%")
            
            with col3:
                total_subs = df['substitutions'].sum()
                st.metric("총 치환 수", f"{total_subs:,}")
            
            with col4:
                total_dels = df['deletions'].sum()
                st.metric("총 삭제 수", f"{total_dels:,}")
            
            with col5:
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