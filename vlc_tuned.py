import librosa
import numpy as np
from scipy import signal
import soundfile as sf
import os

class LibrosaAudioProcessor:
    def __init__(self):
        self.sample_rate = 16000
        
        # VLC 이퀄라이저 설정
        self.eq_bands = {
            60: 5.0,     # 60Hz
            170: 0.7,    # 170Hz
            310: -2.3,   # 310Hz
            600: -0.3,   # 600Hz
            1000: 2.8,   # 1kHz
            3000: -0.7,  # 3kHz
            6000: -4.4,  # 6kHz
            12000: -7.6  # 12kHz
        }
        
        # VLC 음장 효과 설정
        self.room_scale = 0.9  # 크기 (9.0)
        self.width = 0.6       # 너비 (6.0)
        self.wet = 0.2         # 젖은 단계 (2.0)
        self.dry = 0.5         # 건조 단계 (5.0)

    def apply_equalizer(self, y):
        """librosa를 이용한 이퀄라이저 효과"""
        # STFT 변환
        D = librosa.stft(y)
        
        # 주파수 빈 계산
        freqs = librosa.fft_frequencies(sr=self.sample_rate)
        
        # 각 주파수 대역별 게인 적용
        for center_freq, gain_db in self.eq_bands.items():
            # 주파수 대역 범위 설정
            freq_range = center_freq * np.array([0.8, 1.2])  # ±20% 범위
            
            # 해당 주파수 범위의 빈 찾기
            freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
            
            # 게인 적용
            gain_linear = 10 ** (gain_db / 20)
            D[freq_mask] *= gain_linear
        
        # ISTFT로 다시 시간 도메인으로 변환
        return librosa.istft(D, length=len(y))

    def apply_reverb(self, y):
        """리버브 효과 적용"""
        # 리버브 임펄스 응답 생성
        reverb_time = int(self.sample_rate * 0.1)  # 100ms
        impulse = np.exp(-4.0 * np.linspace(0, 1, reverb_time))
        
        # 컨볼루션으로 리버브 적용
        wet = signal.convolve(y, impulse, mode='same')
        
        # 드라이/웨트 믹스
        return y * self.dry + wet * self.wet

def process_audio(input_file, output_file):
    """오디오 파일 처리 메인 함수"""
    try:
        # 오디오 로드
        y, sr = librosa.load(input_file, sr=16000, mono=True)
        
        # 프로세서 초기화
        processor = LibrosaAudioProcessor()
        
        # 이펙트 적용
        y = processor.apply_equalizer(y)
        y = processor.apply_reverb(y)
        
        # 클리핑 방지
        y = np.clip(y, -1.0, 1.0)
        
        # 결과 저장
        sf.write(output_file, y, sr)
        
        print(f"Successfully processed: {input_file} -> {output_file}")
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")

def batch_process(input_dir):
    """디렉토리 일괄 처리"""
    import os
    from pathlib import Path
    
    # 출력 디렉토리 생성
    output_dir = Path(input_dir) / "processed"
    output_dir.mkdir(exist_ok=True)
    
    # 지원 포맷
    supported_formats = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    
    for audio_file in Path(input_dir).glob('*'):
        if audio_file.suffix.lower() in supported_formats:
            output_file = output_dir / f"processed_{audio_file.name}"
            process_audio(str(audio_file), str(output_file))

if __name__ == "__main__":
    input_file = os.getenv("TEST_FILE_NAME"))
    output_file = "output.wav"
    process_audio(input_file, output_file)
    
