from .base import BaseSTTService, TranscriptionResult
import threading
import asyncio
import queue
from typing import Optional, Callable
from .audio_producer import AudioProducer
import traceback

class TranscriptionConsumer(threading.Thread):
    def __init__(self, 
                 audio_producer: AudioProducer,
                 stt_service: BaseSTTService,
                 language_code: str = "ko-KR",
                 on_transcription: Optional[Callable[[str, bool], None]] = None,
                 on_error: Optional[Callable[[Exception], None]] = None,
                 message_queue: Optional[queue.Queue] = None):
        super().__init__()
        self.audio_producer = audio_producer
        self.stt_service = stt_service
        self.language_code = language_code
        self.on_transcription = on_transcription
        self.on_error = on_error
        self.is_running = False
        self.daemon = True
        self.loop = None
        self.message_queue=message_queue
    def print_current_stack(self):
        """현재 스택 트레이스를 출력"""
        print("\n현재 스택 출력:")
        for line in traceback.format_stack():
            print(line.strip())

    async def process_audio_stream(self):
        try:
            await self.stt_service.initialize()

            async def audio_generator():
                while self.is_running:
                    try:
                        chunk = await self.loop.run_in_executor(
                            None, 
                            self.audio_producer.audio_queue.get,
                            True,
                            1
                        )
                        if chunk is None:
                            break
                        yield chunk
                    except queue.Empty:
                        continue

            async for result in self.stt_service.transcribe_stream(
                audio_generator(), 
                self.language_code
            ):
                print("running: " , self.is_running)
                if not self.is_running:
                    break
                print(result) 
                if self.on_transcription:
                    print('call on_transcription', self.on_transcription)
                    self.on_transcription(self.message_queue,result)
                    # await self.loop.run_in_executor(
                    #     None,
                    #     self.on_transcription,
                    #     result.transcript,
                    #     result.is_final
                    # )
            print('existed loop for process_audio_stream')
        except Exception as e:
            print(e)
            if self.on_error:
                await self.loop.run_in_executor(None, self.on_error, e)
        finally:
            await self.stt_service.cleanup()

    def run(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        self.is_running = True
        try:
            self.loop.run_until_complete(self.process_audio_stream())
        finally:
            self.is_running = False
            if self.loop and self.loop.is_running():
                pending = asyncio.all_tasks(self.loop)
                self.loop.run_until_complete(asyncio.gather(*pending))
            if self.loop:
                self.loop.close()

    async def _stop_async(self):
        print('somebody call transcription consumer stop_async')
        self.print_current_stack()
        self.is_running = False

    def stop(self):
        print("somebody call transcription consumer stop")
        self.print_current_stack()
        if self.loop and self.loop.is_running():
            self.loop.create_task(self._stop_async())
        self.is_running = False