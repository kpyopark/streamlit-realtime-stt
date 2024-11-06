import os

import torch
import torchaudio
import numpy as np
from scipy.io import wavfile
from df.enhance import enhance, init_df
import torch.nn.functional as F

class AudioDenoiser:
    def __init__(self):
        # DeepFilterNet 모델 초기화
        self.model, self.state, _ = init_df(
            post_filter=True,
        )
        
    def denoise_audio(self, input_path, output_path):
        """
        음성 파일에서 잡음을 제거하고 원본 샘플링 레이트를 유지합니다.
        
        Args:
            input_path (str): 입력 오디오 파일 경로
            output_path (str): 출력 오디오 파일 경로
        """
        try:
            # 오디오 파일 로드
            waveform, original_sr = torchaudio.load(input_path)
            
            # 스테레오를 모노로 변환 (필요한 경우)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # 모델 처리를 위해 16kHz로 리샘플링
            if original_sr != 16000:
                resampler_down = torchaudio.transforms.Resample(original_sr, 16000)
                waveform_16k = resampler_down(waveform)
            else:
                waveform_16k = waveform
            
            # 잡음 제거 수행 (16kHz에서)
            enhanced_16k = enhance(self.model, self.state, waveform_16k, atten_lim_db=20)
            enhanced_16k_tensor = enhanced_16k # torch.from_numpy(enhanced_16k)
            
            # 원본 샘플링 레이트로 다시 변환
            if original_sr != 16000:
                resampler_up = torchaudio.transforms.Resample(16000, original_sr)
                enhanced = resampler_up(enhanced_16k_tensor)
            else:
                enhanced = enhanced_16k_tensor
            
            # 진폭 정규화
            enhanced = enhanced / torch.max(torch.abs(enhanced))
            
            # int16으로 변환
            #enhanced_int16 = (enhanced * 32767).numpy().astype(np.int16)
            
            # WAV 파일로 저장 (원본 샘플링 레이트 사용)
            #wavfile.write(output_path, int(original_sr), enhanced_int16)
                    # 차원 확인 및 조정
            if enhanced.dim() == 1:
                enhanced = enhanced.unsqueeze(0)  # Add channel dimension
            elif enhanced.dim() == 3:
                enhanced = enhanced.squeeze(0)  # Remove batch dimension if present

            torchaudio.save(output_path, enhanced, int(original_sr))
            print(f"Successfully processed {input_path} -> {output_path} at {original_sr}Hz")
            
        except Exception as e:
            print(f"Error processing {input_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            
            #print("Trying fallback method...")
            #self.denoise_audio_fallback(input_path, output_path)

    def denoise_audio_fallback(self, input_path, output_path):
        """
        기본적인 스펙트럼 서브트랙션을 사용한 대체 방법.
        원본 샘플링 레이트를 유지합니다.
        
        Args:
            input_path (str): 입력 오디오 파일 경로
            output_path (str): 출력 오디오 파일 경로
        """
        try:
            # 오디오 파일 로드
            waveform, original_sr = torchaudio.load(input_path)
            
            # 스테레오를 모노로 변환
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # numpy 배열로 변환
            data = waveform.numpy()
            data = data.flatten()
            
            # 스펙트로그램 계산
            n_fft = min(2048, len(data))
            hop_length = n_fft // 4
            window = np.hanning(n_fft)
            
            # 데이터를 프레임으로 분할
            num_frames = (len(data) - n_fft) // hop_length + 1
            frames = np.zeros((num_frames, n_fft))
            for i in range(num_frames):
                frames[i] = data[i * hop_length:i * hop_length + n_fft] * window
            
            # STFT
            spec = np.fft.rfft(frames, axis=1)
            
            # 노이즈 프로파일 추정 (처음 10프레임 사용)
            noise_profile = np.mean(np.abs(spec[:10]), axis=0)
            
            # 스펙트럼 서브트랙션
            gain = np.maximum(np.abs(spec) - 2 * noise_profile, 0) / (np.abs(spec) + 1e-10)
            spec_enhanced = spec * gain
            
            # ISTFT
            enhanced = np.fft.irfft(spec_enhanced, axis=1)
            
            # 오버랩-애드 합성
            enhanced_signal = np.zeros(len(data))
            window_sum = np.zeros(len(data))
            
            for i in range(num_frames):
                start = i * hop_length
                end = start + n_fft
                enhanced_signal[start:end] += enhanced[i] * window
                window_sum[start:end] += window
            
            # 윈도우 정규화
            mask = window_sum > 1e-10
            enhanced_signal[mask] /= window_sum[mask]
            
            # 진폭 정규화
            enhanced_signal = enhanced_signal / np.max(np.abs(enhanced_signal))
            
            # int16으로 변환
            enhanced_int16 = (enhanced_signal * 32767).astype(np.int16)
            
            # WAV 파일로 저장 (원본 샘플링 레이트 사용)
            wavfile.write(output_path, int(original_sr), enhanced_int16)
            print(f"Successfully processed {input_path} using fallback method -> {output_path} at {original_sr}Hz")
            
        except Exception as e:
            print(f"Error in fallback processing {input_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            
    def process_batch(self, input_files, output_dir):
        """
        여러 오디오 파일을 일괄 처리합니다.
        
        Args:
            input_files (list): 입력 오디오 파일 경로 리스트
            output_dir (str): 출력 디렉토리 경로
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for input_file in input_files:
            filename = os.path.basename(input_file)
            output_path = os.path.join(output_dir, f"denoised_{filename}")
            self.denoise_audio(input_file, output_path)

if __name__ == "__main__":
    denoiser = AudioDenoiser()
    TEST_FILE_NAME = os.getenv("TEST_FILE_NAME")
    denoiser.denoise_audio(TEST_FILE_NAME, "denoised_audio.wav")
    