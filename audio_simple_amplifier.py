import numpy as np
import librosa
import soundfile as sf
from scipy.signal import wiener

class VolumeAmplifier:
    def __init__(self, file_path):
        """
        음량 증폭을 위한 클래스
        
        Args:
            file_path (str): 오디오 파일 경로
        """
        self.audio, self.sr = librosa.load(file_path, sr=None)
        self.original_audio = self.audio.copy()
    
    # def apply_wiener_filter(self, noise_level=0.01, window_size=1001):
    #     """
    #     위너 필터를 사용하여 노이즈 제거
        
    #     Args:
    #         noise_level (float): 예상되는 노이즈 레벨 (0~1 사이 값)
    #         window_size (int): 필터 윈도우 크기 (홀수)
    #     """
    #     # 윈도우 크기가 홀수인지 확인
    #     if window_size % 2 == 0:
    #         window_size += 1
        
    #     # 노이즈 레벨에 따른 mysize 파라미터 조정
    #     mysize = int(noise_level * window_size)
    #     if mysize < 3:
    #         mysize = 3
            
    #     # 위너 필터 적용
    #     self.audio = wiener(self.audio, mysize=mysize)
        
    #     # 필터링 후 정규화
    #     self.audio = self.audio / np.max(np.abs(self.audio))

    def apply_wiener_filter(self, noise_level=0.01, window_size=1001):
        """
        위너 필터를 사용하여 노이즈 제거
        
        Args:
            noise_level (float): 예상되는 노이즈 레벨 (0~1 사이 값)
            window_size (int): 필터 윈도우 크기 (홀수)
        """
        # 윈도우 크기가 홀수인지 확인
        if window_size % 2 == 0:
            window_size += 1
        
        # 노이즈 레벨에 따른 mysize 파라미터 조정
        mysize = int(noise_level * window_size)
        if mysize < 3:
            mysize = 3
        
        # 신호가 너무 작은 경우를 대비한 epsilon 값 설정
        eps = 1e-10
        
        # 신호의 분산이 매우 작은 구간 처리를 위한 사용자 정의 위너 필터
        def custom_wiener(x, mysize, noise=None):
            if noise is None:
                noise = mysize
            
            # 이동 평균과 분산 계산
            window = np.ones(mysize) / mysize
            mean_x = np.convolve(x, window, mode='same')
            mean_x2 = np.convolve(x**2, window, mode='same')
            var_x = mean_x2 - mean_x**2
            
            # 분산이 너무 작은 경우 처리
            var_x = np.maximum(var_x, eps)
            
            # 위너 필터 공식 적용
            noise_var = noise * np.var(x)
            h = np.maximum(1 - noise_var / (var_x + eps), 0)
            
            return mean_x + h * (x - mean_x)
        
        # 필터 적용
        self.audio = custom_wiener(self.audio, mysize)
        
        # 클리핑 방지를 위한 정규화
        max_val = np.max(np.abs(self.audio))
        if max_val > 0:  # 0으로 나누기 방지
            self.audio = self.audio / max_val * 0.99

    def denoise_and_amplify(self, noise_level=0.01, target_loudness_db=-14):
        """
        노이즈 제거 후 스마트 볼륨 증가를 적용하는 통합 메서드
        
        Args:
            noise_level (float): 예상되는 노이즈 레벨 (0~1 사이 값)
            target_loudness_db (float): 목표 음량 레벨 (LUFS와 유사)
        """
        # 1. 노이즈 제거
        self.apply_wiener_filter(noise_level=noise_level)
        
        # 2. 스마트 볼륨 증가
        self.increase_volume_smart(target_loudness_db)

    def simple_gain(self, gain_db):
        """
        단순 게인 증폭 - 가장 기본적인 방법
        
        Args:
            gain_db (float): 증폭할 데시벨 값
        """
        gain_linear = 10 ** (gain_db/20)
        self.audio = self.audio * gain_linear
        
        # 클리핑 체크 및 경고
        max_val = np.max(np.abs(self.audio))
        if max_val > 1.0:
            print(f"경고: 클리핑 발생! 최대값: {max_val}")
            self.audio = np.clip(self.audio, -1.0, 1.0)
    
    def normalize_audio(self, target_db=-3):
        """
        피크 정규화 - 최대 피크를 목표 레벨로 맞춤
        
        Args:
            target_db (float): 목표 피크 레벨 (dB)
        Returns:
            float: 적용된 실제 게인값 (dB)
        """
        current_peak = np.max(np.abs(self.audio))
        current_db = 20 * np.log10(current_peak)
        
        gain = 10**((target_db - current_db) / 20)
        self.audio = self.audio * gain
        
        return 20 * np.log10(gain)
    
    def rms_normalize(self, target_db_rms=-20):
        """
        RMS 정규화 - 전체적인 음량 레벨을 맞춤
        
        Args:
            target_db_rms (float): 목표 RMS 레벨 (dB)
        """
        rms = np.sqrt(np.mean(self.audio**2))
        current_db_rms = 20 * np.log10(rms)
        gain = 10**((target_db_rms - current_db_rms) / 20)
        self.audio = self.audio * gain
    
    def increase_volume_safe(self, target_increase_db=10):
        """
        안전한 볼륨 증가 - 클리핑 방지하며 증폭
        
        Args:
            target_increase_db (float): 목표 증가량 (dB)
        """
        # 현재 피크 레벨 확인
        current_peak = np.max(np.abs(self.audio))
        current_db = 20 * np.log10(current_peak)
        
        # 안전한 증폭 한계 계산
        max_safe_db = -1  # -1dB을 최대 안전 레벨로 설정
        safe_increase = min(target_increase_db, max_safe_db - current_db)
        
        # 증폭 적용
        gain = 10**(safe_increase/20)
        self.audio = self.audio * gain
        
        print(f"안전하게 적용된 증가량: {safe_increase:.1f}dB")
    
    def increase_volume_aggressive(self, target_increase_db=20, max_boost_db=30):
        """
        적극적인 볼륨 증가 - 컴프레션 포함
        
        Args:
            target_increase_db (float): 목표 증가량 (dB)
            max_boost_db (float): 최대 부스트 제한 (dB)
        """
        # 동적 범위 압축 (컴프레션)
        threshold = -20
        ratio = 2
        
        # dB 스케일로 변환
        db = 20 * np.log10(np.abs(self.audio) + 1e-10)
        
        # 컴프레션 적용
        mask = db > threshold
        compressed = np.copy(self.audio)
        compressed[mask] = np.sign(self.audio[mask]) * (
            10**((threshold + (db[mask] - threshold) / ratio) / 20)
        )
        
        # 메이크업 게인 적용
        gain = min(target_increase_db, max_boost_db)
        gain_linear = 10**(gain/20)
        
        self.audio = compressed * gain_linear
        
        # 피크 제한
        self.audio = np.clip(self.audio, -0.99, 0.99)
    
    def analyze_volume(self):
        """
        현재 오디오의 볼륨 정보 분석
        
        Returns:
            dict: 볼륨 관련 정보
        """
        peak = np.max(np.abs(self.audio))
        rms = np.sqrt(np.mean(self.audio**2))
        
        return {
            'peak_db': 20 * np.log10(peak),
            'rms_db': 20 * np.log10(rms),
            'dynamic_range_db': 20 * np.log10(peak/rms),
            'is_clipping': peak >= 1.0
        }
    
    def increase_volume_smart(self, target_loudness_db=-14):
        """
        스마트 볼륨 증가 - 여러 방법을 조합
        
        Args:
            target_loudness_db (float): 목표 음량 레벨 (LUFS와 유사)
        """
        # 1. 현재 음량 분석
        analysis = self.analyze_volume()
        
        # 2. 동적 범위가 큰 경우 가벼운 컴프레션 적용
        if analysis['dynamic_range_db'] > 20:
            self.increase_volume_aggressive(target_increase_db=10, max_boost_db=15)
        
        # 3. RMS 기반 정규화
        self.rms_normalize(target_db_rms=target_loudness_db)
        
        # 4. 피크 제한
        peak = np.max(np.abs(self.audio))
        if peak > 0.99:
            self.audio = self.audio * (0.99 / peak)
    
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

if __name__ == "__main__":
    TEST_FILE_NAME = os.getenv("TEST_FILE_NAME")
    denoiser = VolumeAmplifier(TEST_FILE_NAME)
    #denoiser.apply_wiener_filter(noise_level=0.01)
    denoiser.denoise_and_amplify(noise_level=0.02, target_loudness_db=-13)
    denoiser.save("denoised_audio_amplified.wav")
