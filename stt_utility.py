import numpy as np
import numpy as np
from itertools import zip_longest
import re
import unicodedata
import os
from pydub import AudioSegment
import io
from pathlib import Path



def convert_to_wav(audio_file, SAMPLING_RATE):
    try:
        original_filename = os.path.basename(audio_file.name)
        filename_without_ext = os.path.splitext(original_filename)[0]
        output_filename = f"{filename_without_ext}_{SAMPLING_RATE}.wav"
        file_path = Path(output_filename)
        if file_path.exists():
            return output_filename
        audio = AudioSegment.from_file(audio_file)
        #audio = audio.set_channels(NCHANNEL)
        audio = audio.set_frame_rate(SAMPLING_RATE)
        wav_buffer = io.BytesIO()
        audio.export(
            wav_buffer,
            format="wav",
            parameters=["-q:a", "0"]  # 최고 품질 설정
        )
        with open(output_filename, 'wb') as f:
            f.write(wav_buffer.getvalue())
        return output_filename
    except Exception as e:
        print(e)
        return None
    
def clean_chinese_text(text):
    """
    중국어 텍스트에서 한자, 영문자, 숫자를 제외한 문자 제거
    
    Args:
        text (str): 원본 텍스트
    
    Returns:
        str: 정제된 텍스트
    """
    # 1. 모든 공백 제거
    text = re.sub(r'\s+', '', text)
    
    # 2. 한자 범위 정의 (CJK Unified Ideographs)
    chinese_char_ranges = [
        (0x4E00, 0x9FFF),   # CJK Unified Ideographs
        (0x3400, 0x4DBF),   # CJK Unified Ideographs Extension A
        (0x20000, 0x2A6DF), # CJK Unified Ideographs Extension B
        (0x2A700, 0x2B73F), # CJK Unified Ideographs Extension C
        (0x2B740, 0x2B81F), # CJK Unified Ideographs Extension D
        (0x2B820, 0x2CEAF), # CJK Unified Ideographs Extension E
        (0x2CEB0, 0x2EBEF), # CJK Unified Ideographs Extension F
        (0x30000, 0x3134F), # CJK Unified Ideographs Extension G
        (0xF900, 0xFAFF),   # CJK Compatibility Ideographs
    ]
    
    def is_valid_char(char):
        # 한자 체크
        code = ord(char)
        is_chinese = any(start <= code <= end for start, end in chinese_char_ranges)
        
        # 영문자 체크 (대소문자)
        is_english = char.isascii() and char.isalpha()
        
        # 숫자 체크
        is_number = char.isdigit()
        
        return is_chinese or is_english or is_number
    
    # 3. 유효한 문자만 유지
    cleaned_text = ''.join(char.lower() for char in text if is_valid_char(char))
    
    return cleaned_text

def compute_cer(reference, hypothesis, preprocess=True):
    """
    중국어 Character Error Rate 계산
    
    Args:
        reference (str): 정답 텍스트
        hypothesis (str): 인식/전사된 텍스트
        preprocess (bool): 전처리 수행 여부
    
    Returns:
        float: CER 값 (0~1)
        dict: 상세 통계
    """
    # 전처리 수행
    if preprocess:
        original_reference = reference
        original_hypothesis = hypothesis
        reference = clean_chinese_text(reference)
        hypothesis = clean_chinese_text(hypothesis)
    
    # 동적 프로그래밍을 위한 행렬 초기화
    ref_len = len(reference) + 1
    hyp_len = len(hypothesis) + 1
    matrix = np.zeros((ref_len, hyp_len))
    
    # 행렬 초기값 설정
    for i in range(ref_len):
        matrix[i, 0] = i
    for j in range(hyp_len):
        matrix[0, j] = j
    
    # 작업 기록을 위한 백트래스 행렬
    operations = np.zeros((ref_len, hyp_len), dtype=str)
    operations[0,1:] = 'I'
    operations[1:,0] = 'D'
    
    # 편집 거리 계산
    for i in range(1, ref_len):
        for j in range(1, hyp_len):
            if reference[i-1] == hypothesis[j-1]:
                matrix[i, j] = matrix[i-1, j-1]
                operations[i, j] = 'M'  # Match
            else:
                substitution = matrix[i-1, j-1] + 1
                deletion = matrix[i-1, j] + 1
                insertion = matrix[i, j-1] + 1
                
                min_dist = min(substitution, deletion, insertion)
                matrix[i, j] = min_dist
                
                if min_dist == substitution:
                    operations[i, j] = 'S'
                elif min_dist == deletion:
                    operations[i, j] = 'D'
                else:
                    operations[i, j] = 'I'
    
    # 오류 카운트
    i, j = len(reference), len(hypothesis)
    substitutions = deletions = insertions = 0
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and operations[i,j] == 'S':
            substitutions += 1
            i -= 1
            j -= 1
        elif i > 0 and operations[i,j] == 'D':
            deletions += 1
            i -= 1
        elif j > 0 and operations[i,j] == 'I':
            insertions += 1
            j -= 1
        else:  # Match
            i -= 1
            j -= 1
    
    total_errors = substitutions + deletions + insertions
    cer = total_errors / len(reference)
    
    # 문자별 비교를 위한 정렬된 시퀀스 생성
    aligned_ref = []
    aligned_hyp = []
    i, j = len(reference), len(hypothesis)
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and operations[i,j] == 'M':
            aligned_ref.insert(0, reference[i-1])
            aligned_hyp.insert(0, hypothesis[j-1])
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and operations[i,j] == 'S':
            aligned_ref.insert(0, reference[i-1])
            aligned_hyp.insert(0, hypothesis[j-1])
            i -= 1
            j -= 1
        elif i > 0 and operations[i,j] == 'D':
            aligned_ref.insert(0, reference[i-1])
            aligned_hyp.insert(0, '∅')
            i -= 1
        else:  # Insertion
            aligned_ref.insert(0, '∅')
            aligned_hyp.insert(0, hypothesis[j-1])
            j -= 1
    
    # 문자별 비교 정보 생성
    comparison = []
    for ref_char, hyp_char in zip(aligned_ref, aligned_hyp):
        if ref_char == hyp_char:
            comparison.append(f"{ref_char}→{hyp_char} (일치)")
        elif hyp_char == '∅':
            comparison.append(f"{ref_char}→∅ (삭제)")
        elif ref_char == '∅':
            comparison.append(f"∅→{hyp_char} (삽입)")
        else:
            comparison.append(f"{ref_char}→{hyp_char} (치환)")
    
    stats = {
        'CER': round(cer * 100, 2),
        'Total Errors': total_errors,
        'Substitutions': substitutions,
        'Deletions': deletions,
        'Insertions': insertions,
        'Reference Length': len(reference),
        'Character Comparison': comparison
    }
    
    # 전처리 정보 추가
    if preprocess:
        stats['Original Reference'] = original_reference
        stats['Cleaned Reference'] = reference
        stats['Original Hypothesis'] = original_hypothesis
        stats['Cleaned Hypothesis'] = hypothesis
        stats['Removed Characters'] = {
            'Reference': ''.join(char for char in original_reference if char not in reference),
            'Hypothesis': ''.join(char for char in original_hypothesis if char not in hypothesis)
        }
    
    return cer, stats

def extract_transcript_from_googlestt_result(json_data):
    """
    JSON 데이터에서 transcript 텍스트만 추출하는 함수
    
    Args:
        json_data (str or dict): JSON 문자열 또는 딕셔너리
        
    Returns:
        str: 추출된 transcript 텍스트
    """
    # 문자열로 입력된 경우 JSON 파싱
    if isinstance(json_data, str):
        data = json.loads(json_data)
    else:
        data = json_data
        
    # transcript 추출
    try:
        transcripts = []
        for result in data['results']:
            if 'alternatives' in result:
                alternative = result['alternatives'][0]
                if 'transcript' in alternative:
                    transcripts.append(alternative['transcript'])
        
        # 추출된 transcript들을 공백으로 구분하여 결합
        combined_transcript = ' '.join(transcripts)
        return combined_transcript
    except (KeyError, IndexError) as e:
        print(e)
        return f"Error: Unable to extract transcript - {str(e)}"

def test_extract_transcript_from_googlestt_result():
    sample_json = {
        "results": [
            {
                "alternatives": [{
                    "transcript": "第一段现象检查记忆叠加不齐原电池变形...",
                    "confidence": 0.90555483,
                    "words": [...]
                }]
            },
            {
                "alternatives": [{
                    "transcript": "另一段文字内容...",
                    "confidence": 0.92555483,
                    "words": [...]
                }]
            }
        ]
    }    
    result = extract_transcript_from_googlestt_result(sample_json)
    print("추출된 텍스트:", result)

def test_compute_cer():
    reference = "今天(星期一)的天气Temp23C,真的不错!!"
    hypothesis = "今天 (星期一) 的天气temp23c，真的 不错！"
    
    cer, stats = compute_cer(reference, hypothesis, preprocess=True)
    
    print("원본 텍스트:")
    print(f"참조: {reference}")
    print(f"인식: {hypothesis}")
    
    print("\n전처리 결과:")
    print(f"정제된 참조: {stats['Cleaned Reference']}")
    print(f"정제된 인식: {stats['Cleaned Hypothesis']}")
    print(f"\n제거된 문자:")
    print(f"참조: {stats['Removed Characters']['Reference']}")
    print(f"인식: {stats['Removed Characters']['Hypothesis']}")
    
    print("\n상세 통계:")
    for k, v in stats.items():
        if k not in ['Character Comparison', 'Original Reference', 'Original Hypothesis', 
                    'Cleaned Reference', 'Cleaned Hypothesis', 'Removed Characters']:
            print(f"{k}: {v}")
    
    print("\n문자별 비교:")
    for comp in stats['Character Comparison']:
        print(comp)

# 테스트
if __name__ == "__main__":
    test_extract_transcript_from_googlestt_result()