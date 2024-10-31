
def convert_to_wav(audio_file):
    try:
        original_filename = os.path.basename(audio_file.name)
        filename_without_ext = os.path.splitext(original_filename)[0]
        output_filename = f"{filename_without_ext}_{SAMPLING_RATE}.wav"
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
        st.error(f"오디오 변환 중 오류가 발생했습니다: {str(e)}")
        return None
    
