import os
from dotenv import load_dotenv

from google.cloud.speech_v2 import SpeechClient, SpeechAsyncClient
from google.cloud.speech_v2.types import cloud_speech
from google.api_core.client_options import ClientOptions

from pydub import AudioSegment
import io
import asyncio
import time
from queue import Queue, Empty
import threading
import traceback
import wave
import stt_utility
from pathlib import Path


load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = 'us'
TEST_FILE_NAME = os.getenv("TEST_FILE_NAME")
SAMPLING_RATE = int(os.getenv("SAMPLING_RATE"))

def convert_to_wav(audio_file, SAMPLING_RATE):
    try:
        original_filename = os.path.basename(audio_file)
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

class AudioFeeder(threading.Thread):
    def __init__(self, 
        wav_data: bytes, 
        queue: Queue,
        sampling_rate: int = 16000,
        num_of_chunks_in_a_sec: int = 20,
        feeding_segment_window: int = 1, 
        need_header: bool = False):
        super().__init__()
        self.wav_data = wav_data
        self.queue = queue
        self.sampling_rate = sampling_rate
        self.num_of_chunks_in_a_sec = num_of_chunks_in_a_sec
        self.chunk_duration_ms = 1000 // self.num_of_chunks_in_a_sec
        self.feeding_segment_window = feeding_segment_window
        self.need_header = need_header
        self.is_playing = False
        self.wf = wave.open(io.BytesIO(self.wav_data), 'rb')
        self.wf_param = self.wf.getparams()
        self.content = self.wf.readframes(self.wf.getnframes())
        self.chunk_length = ( self.sampling_rate * self.wf_param.nchannels * self.wf_param.sampwidth ) // self.num_of_chunks_in_a_sec
        #print(self.content)
        print(self.chunk_length)
        print(self.chunk_duration_ms)

    def run(self):
        try:
            self.is_playing = True
            start_time = time.time()
            cnt = 0        
            # 오디오 데이터를 청크로 나누어 큐에 넣기
            for chunk_index in range(0, len(self.content) // self.chunk_length):
                start_position = chunk_index * self.chunk_length
                chunk = self.content[start_position:start_position + self.chunk_length]
                
                # 실제 시간과 경과 시간을 계산하여 적절한 딜레이 추가
                actual_time = (chunk_index * self.chunk_duration_ms) / 1000
                elapsed_time = time.time() - start_time
                wait_time = max(0, actual_time - elapsed_time)
                
                if wait_time > 0:
                    time.sleep(wait_time)
                self.flush(chunk)
                cnt = cnt + 1
                if cnt % self.num_of_chunks_in_a_sec == 0:
                    print(f'feeding chunk - {cnt // self.num_of_chunks_in_a_sec} seconds')

                #print(self.is_playing)                
                if not self.is_playing:
                    break
        except Exception as e :
            print('error: ', e)
            print(traceback.format_exc())

        finally:
            # 피딩 완료 표시
            self.is_playing = False
            self.flush(None)
            print("Audio feeding completed")
            self.wf.close()
    
    def flush(self, chunk):
        if self.need_header:
            with io.BytesIO() as wav_buffer:
                with wave.open(wav_buffer, 'wb') as output_buffer:
                    output_buffer.setparams(self.wf_param)
                    output_buffer.writeframes(chunk)
                    try:
                        self.queue.put(wav_buffer.getvalue(), block=False)
                    except queue.Full:
                        print("Warning: Audio queue is full, skipping chunk")
            self.queue.put(chunk)
        else:
            self.queue.put(chunk)

    def stop(self):
        """Stop audio playback and streaming"""
        self.is_playing = False

class ResumableStreamGenerator:
    def __init__(self, 
        config: cloud_speech.StreamingRecognizeRequest,
        audio_queue: Queue,
        num_of_chunks_in_a_sec: int = 20, 
        num_of_overwrapped_chunks: int = 4):
        self.config = config
        self.audio_queue = audio_queue
        self.num_of_chunks_in_a_sec = num_of_chunks_in_a_sec
        self.num_of_overwrapped_chunks = num_of_overwrapped_chunks
        self.chunk_duration_ms = 1000 // num_of_chunks_in_a_sec
        self.need_resume = False
        self.need_close = False
        self.unprocessed_audio_chunks = []
        self.last_processed_time_ms = 0.0
        self.total_processed_time_ms = 0.0
        self.feeding_time_ms = 0.0
        self.current_stream_start_time = time.time()
    
    def notify_last_transcription(self, response):
        if self.need_resume:
            # 새로 resume이 필요한 경우는 신규 generator가 만들어질 때까지 그냥 처리하지 않는 것으로.
            return
        # 동일 result에서는 처리 시간이 동일 한개만 처리하면 됨. 
        if not response.results:
            return
        result = response.results[0]
        if not result.alternatives:
            return
        if result.result_end_offset:
            result_end_offset = result.result_end_offset
            last_processed_time_ms = result_end_offset.total_seconds() * 1000
            total_last_processed_time_ms = self.total_processed_time_ms + last_processed_time_ms
            session_duration_sec = time.time() - self.current_stream_start_time
            # print('feeding time ms                    :', self.feeding_time_ms)
            # print('last_processed_time_ms in this ses.:', last_processed_time_ms)
            # print('transcribe end                     :', total_last_processed_time_ms)
            # print('gap between feeding and processing :', (self.feeding_time_ms - total_last_processed_time_ms) )
            if (session_duration_sec > 30) and result.is_final:
                ## Overwrap Chunk를 위한 시간 조정
                last_processed_time_ms = last_processed_time_ms - self.chunk_duration_ms * self.num_of_overwrapped_chunks
                processed_num_of_chunks = int(last_processed_time_ms // self.chunk_duration_ms)
                print('processed_gap_time       :', last_processed_time_ms)
                print('audio_chunks in this ses.: ', len(self.unprocessed_audio_chunks))
                self.unprocessed_audio_chunks = self.unprocessed_audio_chunks[processed_num_of_chunks:]
                self.last_processed_time_ms = (last_processed_time_ms // self.chunk_duration_ms) * self.chunk_duration_ms
                self.need_resume = True
                print('unprocessed_audio_chunks : ', len(self.unprocessed_audio_chunks))
                print('last_processed_time_ms   : ', self.last_processed_time_ms)

        
    def generator(self) -> list:
        # 설정 요청 먼저 전송
        self.total_processed_time_ms = self.total_processed_time_ms + self.last_processed_time_ms
        self.current_stream_start_time = time.time()
        self.need_resume = False
        print('total_processed_time_ms  :', self.total_processed_time_ms)
        print('feeding_time_ms          :', self.feeding_time_ms)
        yield self.config
        cnt = 0
        self.feeding_time_ms = self.total_processed_time_ms
        # 먼저 잔여분을 전송시킴
        for chunk in self.unprocessed_audio_chunks:
            self.feeding_time_ms = self.feeding_time_ms + self.chunk_duration_ms
            yield chunk
        # 큐에서 오디오 청크를 가져와서 전송
        while not self.need_resume:
            try:
                audio_chunk = self.audio_queue.get(timeout=0.01)
                if audio_chunk is None:
                    print('None is received.')
                    self.need_close = True
                    break
                self.feeding_time_ms = self.feeding_time_ms + self.chunk_duration_ms
                cnt = cnt + 1
                if cnt % self.num_of_chunks_in_a_sec == 0 :
                    print(f'feeded chunks of {self.feeding_time_ms} ms.')
                audio_request = cloud_speech.StreamingRecognizeRequest(audio=audio_chunk)
                self.unprocessed_audio_chunks.append(audio_request)
                yield audio_request
            except Empty:
                continue
            except Exception as e :
                print ('error in generator:', e)
                print(traceback.format_exc())
        if self.need_resume:
            print("need restart generator.")
        print("Exit Generator")
        

def transcribe_streaming_v2_sync(
    audio_file: str,
    location: str,
    model: str,
    language_code: str
) -> cloud_speech.StreamingRecognizeResponse:

    output_file_name = convert_to_wav(audio_file, SAMPLING_RATE)
    audio = AudioSegment.from_file(output_file_name)
    wav_buffer = io.BytesIO()
    audio.export(wav_buffer, format="wav")
    wav_data = wav_buffer.getvalue()

    num_of_chunks_in_a_sec = 10
    chunk_length = SAMPLING_RATE // num_of_chunks_in_a_sec
    chunk_duration_ms = 1000 // num_of_chunks_in_a_sec

    # 오디오 청크를 저장할 큐 생성
    audio_queue = Queue()
    
    audio_feeder = AudioFeeder(
        wav_data = wav_data,
        queue = audio_queue,
        sampling_rate = SAMPLING_RATE,
        num_of_chunks_in_a_sec = num_of_chunks_in_a_sec,
        feeding_segment_window = 1
    )

    audio_feeder.start()

    # Instantiates a client
    if location == 'global':
        client = SpeechClient()
    else:
        client = SpeechClient(
            client_options=ClientOptions(
                api_endpoint=f"{location}-speech.googleapis.com",
            )
        )

    # Recognition 설정
    recognition_config = cloud_speech.RecognitionConfig(
        explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
            sample_rate_hertz=SAMPLING_RATE,
            encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
            audio_channel_count=1
        ),
        language_codes=[language_code],
        model=model,
    )
    
    streaming_config = cloud_speech.StreamingRecognitionConfig(
        config=recognition_config,
        streaming_features=cloud_speech.StreamingRecognitionFeatures(
            interim_results=True
        ),
    )
    
    config_request = cloud_speech.StreamingRecognizeRequest(
        recognizer=f"projects/{PROJECT_ID}/locations/{location}/recognizers/_",
        streaming_config=streaming_config,
    )

    # Transcription 실행
    responses = []
    print("Starting recognition...")
    transcripts_full = []
    try:
        stream = ResumableStreamGenerator(
            config_request, 
            audio_queue,
            num_of_chunks_in_a_sec
        )
        while not stream.need_close:
            responses_iterator = client.streaming_recognize(
                #requests=request_generator(config_request)
                requests=stream.generator()
            )
            
            for response in responses_iterator:
                #print("Response received.")
                responses.append(response)
                stream.notify_last_transcription(response)
                if not response.results:
                    continue
                transcript = ""
                is_final = "F"
                for result in response.results:
                    if not result.alternatives:
                        continue
                    for alternative in result.alternatives:
                        transcript = transcript + alternative.transcript
                    if result.is_final:
                        is_final = "T"
                        transcripts_full.append(alternative.transcript)
                    #print(result)
                print(f'transcript ({is_final}) :', transcript)
        print("response_iterator is finised.")
    except Exception as e:
        print('error in making request: ', e)
        print(traceback.format_exc())
    finally:
        # 쓰레드 종료 대기
        audio_feeder.stop()
        audio_feeder.join()
    print('\n'.join(transcripts_full))
    return responses

def sync_main():
    print(TEST_FILE_NAME)
    transcribe_streaming_v2_sync(TEST_FILE_NAME, "us", "long", "cmn-Hans-CN") ### OK

if __name__ == "__main__":
    sync_main()