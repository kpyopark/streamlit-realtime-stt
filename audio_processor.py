import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt
import librosa
import soundfile as sf

class AudioProcessor:
    def __init__(self, file_path):
        """
        오디오 파일을 로드하고 처리하는 클래스
        
        Args:
            file_path (str): 오디오 파일 경로
        """
        self.audio, self.sr = librosa.load(file_path, sr=None)
                # 기본 주파수 대역 설정 (Hz)
        self.frequency_bands = {
            'Sub Bass': (20, 60),
            'Bass': (60, 250),
            'Low Mids': (250, 500),
            'Mids': (500, 2000),
            'High Mids': (2000, 4000),
            'Presence': (4000, 6000),
            'Brilliance': (6000, 20000)
        }
        
    def design_band_filter(self, low_freq, high_freq, order=4):
        """
        특정 주파수 대역의 밴드패스 필터 설계
        
        Args:
            low_freq (float): 하한 주파수
            high_freq (float): 상한 주파수
            order (int): 필터 차수
        
        Returns:
            tuple: 필터 계수 (b, a)
        """
        nyquist = self.sr * 0.5
        low = low_freq / nyquist
        high = high_freq / nyquist
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def apply_equalizer(self, gains_db):
        """
        각 주파수 대역별로 게인을 적용하는 이퀄라이저
        
        Args:
            gains_db (dict): 각 대역별 게인값 (dB)
                예: {
                    'Sub Bass': -3,
                    'Bass': 2,
                    'Low Mids': 0,
                    'Mids': 1,
                    'High Mids': 2,
                    'Presence': 3,
                    'Brilliance': -1
                }
        """
        # 원본 오디오 복사
        processed_audio = np.zeros_like(self.audio)
        
        # 각 주파수 대역별로 처리
        for band_name, (low_freq, high_freq) in self.frequency_bands.items():
            # 해당 대역의 게인값 가져오기
            gain_db = gains_db.get(band_name, 0)
            gain_linear = 10 ** (gain_db / 20)
            
            # 밴드패스 필터 설계 및 적용
            b, a = self.design_band_filter(low_freq, high_freq)
            band_audio = filtfilt(b, a, self.audio)
            
            # 게인 적용 및 결과 누적
            processed_audio += band_audio * gain_linear
        
        # 결과 저장
        self.audio = processed_audio
        
        # 클리핑 방지
        max_val = np.max(np.abs(self.audio))
        if max_val > 1.0:
            self.audio = self.audio / max_val

    def remove_noise(self, cutoff_freq=1000, order=5):
        """
        저역통과 필터를 사용하여 고주파 잡음을 제거
        
        Args:
            cutoff_freq (int): 차단 주파수 (Hz)
            order (int): 필터 차수
        """
        nyquist = self.sr * 0.5
        normalized_cutoff_freq = cutoff_freq / nyquist
        b, a = butter(order, normalized_cutoff_freq, btype='low', analog=False)
        self.audio = filtfilt(b, a, self.audio)
        
    def spectral_noise_reduction(self, noise_clip, reduce_factor=0.7):
        """
        스펙트럼 감산을 통한 잡음 제거
        
        Args:
            noise_clip (array): 잡음 샘플
            reduce_factor (float): 잡음 감소 강도 (0-1)
        """
        # 잡음 스펙트럼 추정
        noise_stft = librosa.stft(noise_clip)
        noise_power = np.mean(np.abs(noise_stft)**2, axis=1)
        
        # 신호 스펙트럼
        audio_stft = librosa.stft(self.audio)
        audio_power = np.abs(audio_stft)**2
        
        # 잡음 제거
        mask = (audio_power - reduce_factor * noise_power.reshape(-1, 1)) / audio_power
        mask = np.maximum(mask, 0)
        self.audio = librosa.istft(audio_stft * mask)
        
    def amplify_voice(self, gain_db=10):
        """
        음성 신호 증폭
        
        Args:
            gain_db (float): 증폭할 데시벨 값
        """
        gain_linear = 10**(gain_db/20)
        self.audio = self.audio * gain_linear
        
        # 클리핑 방지
        max_val = np.max(np.abs(self.audio))
        if max_val > 1.0:
            self.audio = self.audio / max_val
            
    def apply_compression(self, threshold_db=-20, ratio=4):
        """
        다이나믹 레인지 압축을 적용하여 소리를 더 균일하게 만듦
        
        Args:
            threshold_db (float): 압축 시작 임계값 (dB)
            ratio (float): 압축 비율
        """
        # dB로 변환
        db = 20 * np.log10(np.abs(self.audio) + 1e-10)
        
        # 압축 적용
        mask = db > threshold_db
        compressed = np.copy(self.audio)
        compressed[mask] = np.sign(self.audio[mask]) * (
            10**((threshold_db + (db[mask] - threshold_db) / ratio) / 20)
        )
        self.audio = compressed
        
    def save(self, output_path):
        """
        처리된 오디오를 파일로 저장
        
        Args:
            output_path (str): 저장할 파일 경로
        """
        sf.write(output_path, self.audio, self.sr)