import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, filtfilt

class AudioAmplifier:
    def __init__(self, file_path):
        """
        오디오 파일을 로드하고 증폭하는 클래스
        
        Args:
            file_path (str): 오디오 파일 경로
        """
        self.audio, self.sr = librosa.load(file_path, sr=None)
        self.original_audio = self.audio.copy()
        
    def normalize_audio(self, target_db=-1):
        """
        오디오를 목표 데시벨로 정규화
        
        Args:
            target_db (float): 목표 피크 레벨 (dB)
        """
        # 현재 피크 레벨 계산
        current_peak = np.max(np.abs(self.audio))
        current_db = 20 * np.log10(current_peak)
        
        # 목표 레벨까지 증폭
        gain = 10**((target_db - current_db) / 20)
        self.audio = self.audio * gain
        
    def amplify_with_limiter(self, gain_db=20, threshold_db=-1):
        """
        리미터를 사용하여 클리핑 없이 증폭
        
        Args:
            gain_db (float): 증폭할 데시벨 값
            threshold_db (float): 리미터 임계값 (dB)
        """
        # 선형 게인으로 변환
        gain_linear = 10**(gain_db/20)
        
        # 증폭
        amplified = self.audio * gain_linear
        
        # 리미터 적용
        threshold_linear = 10**(threshold_db/20)
        max_val = np.max(np.abs(amplified))
        
        if max_val > threshold_linear:
            ratio = threshold_linear / max_val
            amplified = amplified * ratio
            
        self.audio = amplified
        
    def apply_makeup_gain(self, target_lufs=-14):
        """
        LUFS 기반 메이크업 게인 적용
        
        Args:
            target_lufs (float): 목표 LUFS 레벨
        """
        # 현재 LUFS 레벨 측정
        current_lufs = librosa.feature.rms(y=self.audio).mean()
        current_lufs_db = 20 * np.log10(current_lufs + 1e-10)
        
        # 필요한 게인 계산 및 적용
        gain_db = target_lufs - current_lufs_db
        self.audio = self.audio * (10 ** (gain_db/20))
        
    def enhance_clarity(self, boost_db=6):
        """
        중요 주파수 대역 부스트로 명료도 향상
        
        Args:
            boost_db (float): 부스트할 데시벨 값
        """
        # 음성 주파수 대역 (1kHz - 4kHz) 부스트
        nyquist = self.sr * 0.5
        low_freq = 1000 / nyquist
        high_freq = 4000 / nyquist
        
        b, a = butter(4, [low_freq, high_freq], btype='band')
        voice_band = filtfilt(b, a, self.audio)
        
        # 부스트 적용
        boost_factor = 10**(boost_db/20)
        self.audio = self.audio + (voice_band * (boost_factor - 1))
        
    def apply_dynamic_compression(self, threshold_db=-20, ratio=4, attack_ms=5, release_ms=50):
        """
        다이나믹 컴프레서 적용
        
        Args:
            threshold_db (float): 압축 시작 임계값 (dB)
            ratio (float): 압축 비율
            attack_ms (float): 어택 시간 (ms)
            release_ms (float): 릴리즈 시간 (ms)
        """
        # 시간 상수 계산
        attack_samples = int(attack_ms * self.sr / 1000)
        release_samples = int(release_ms * self.sr / 1000)
        
        # 게인 리덕션 계산
        db = 20 * np.log10(np.abs(self.audio) + 1e-10)
        gain_reduction = np.zeros_like(db)
        mask = db > threshold_db
        gain_reduction[mask] = (db[mask] - threshold_db) * (1 - 1/ratio)
        
        # 시간 상수 적용
        smoothed_gain = np.zeros_like(gain_reduction)
        for i in range(1, len(gain_reduction)):
            if gain_reduction[i] < smoothed_gain[i-1]:
                coeff = 1 - np.exp(-1/attack_samples)
            else:
                coeff = 1 - np.exp(-1/release_samples)
            smoothed_gain[i] = coeff * gain_reduction[i] + (1-coeff) * smoothed_gain[i-1]
        
        # 압축 적용
        gain_linear = 10**(-smoothed_gain/20)
        self.audio = self.audio * gain_linear
        
    def smart_amplify(self, target_loudness_db=-14, clarity_boost=True):
        """
        스마트 증폭 - 여러 방법을 조합하여 최적의 결과 도출
        
        Args:
            target_loudness_db (float): 목표 음량 레벨 (dB)
            clarity_boost (bool): 명료도 향상 적용 여부
        """
        # 1. 기본 정규화
        self.normalize_audio(target_db=20)
        
        # 2. 다이나믹 컴프레션 적용
        self.apply_dynamic_compression(threshold_db=-20, ratio=3)
        
        # 3. 명료도 향상 (선택사항)
        if clarity_boost:
            self.enhance_clarity(boost_db=4)
        
        # 4. 메이크업 게인으로 최종 음량 조정
        self.apply_makeup_gain(target_lufs=target_loudness_db)
        
        # 5. 리미터로 피크 제어
        self.amplify_with_limiter(threshold_db=-1)
    
    def save(self, output_path):
        """
        처리된 오디오를 파일로 저장
        
        Args:
            output_path (str): 저장할 파일 경로
        """
        sf.write(output_path, self.audio, self.sr)
        
    def reset(self):
        """
        오디오를 원본 상태로 리셋
        """
        self.audio = self.original_audio.copy()