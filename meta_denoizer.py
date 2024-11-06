import torch
import torchaudio
from demucs.pretrained import get_pretrained
from demucs.apply import apply_model
import os
import numpy as np
from pathlib import Path

class AudioSeparator:
    def __init__(self, model_name='htdemucs', device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the audio separator
        Args:
            model_name (str): Name of the Demucs model to use
            device (str): Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device
        self.model = get_pretrained(model_name)
        self.model.to(device)
        self.sample_rate = 44100  # Demucs default sample rate
        
    def load_audio(self, audio_path):
        """
        Load audio file and resample if necessary
        Args:
            audio_path (str): Path to the audio file
        Returns:
            torch.Tensor: Audio tensor
        """
        waveform, sr = torchaudio.load(audio_path)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        return waveform

    def separate_audio(self, audio_path, output_dir):
        """
        Separate audio into different stems
        Args:
            audio_path (str): Path to input audio file
            output_dir (str): Directory to save separated audio files
        Returns:
            dict: Paths to separated audio files
        """
        # Create output directory if it doesn't exist
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and process audio
        wav = self.load_audio(audio_path)
        
        # Apply separation model
        sources = apply_model(self.model, wav, device=self.device)
        sources = sources.cpu().numpy()
        
        # Get source names from model
        source_names = self.model.sources
        
        # Save separated sources
        output_paths = {}
        for source, name in zip(sources, source_names):
            source = torch.from_numpy(source)
            output_path = output_dir / f"{name}.wav"
            torchaudio.save(
                output_path,
                source,
                self.sample_rate
            )
            output_paths[name] = str(output_path)
            
        return output_paths

    def extract_vocals(self, audio_path, output_dir):
        """
        Extract only the vocals from the audio
        Args:
            audio_path (str): Path to input audio file
            output_dir (str): Directory to save vocal track
        Returns:
            str: Path to extracted vocals file
        """
        separated = self.separate_audio(audio_path, output_dir)
        return separated.get('vocals')

def main():
    # Example usage
    separator = AudioSeparator()
    
    # 입력 파일과 출력 디렉토리 설정
    input_file = os.getenv("TEST_FILE_NAME")  # 실제 입력 파일 경로로 변경하세요
    output_dir = "separated_audio"   # 실제 출력 디렉토리로 변경하세요
    
    try:
        # 전체 분리 실행
        separated_paths = separator.separate_audio(input_file, output_dir)
        print("분리된 오디오 파일 경로:")
        for source, path in separated_paths.items():
            print(f"{source}: {path}")
            
        # 음성만 추출
        vocals_path = separator.extract_vocals(input_file, output_dir)
        print(f"\n추출된 음성 파일 경로: {vocals_path}")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")

if __name__ == "__main__":
    main()