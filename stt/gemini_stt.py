from .base import BaseSTTService, TranscriptionResult
from google.cloud import speech_v2 as speech
from google.api_core.client_options import ClientOptions
import os
from typing import AsyncGenerator
from dotenv import load_dotenv
import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part, SafetySetting
import json
import re

load_dotenv()

class GeminiSTTService(BaseSTTService):
    def __init__(self):
        self.project_id = os.getenv("PROJECT_ID")
        self.location = os.getenv("LOCATION")
        self.isProgressing = False
        self.last_3_transcripts = []
        self.last_3_wav_chunks = []

    async def initialize(self):
        vertexai.init(project=self.project_id, location=self.location)
        self.gemini_model = GenerativeModel(
          "gemini-1.5-flash-002",
          system_instruction=["""You are a simultaneous interpreter working at a Chinese battery manufacturing plant for electric vehicles. Your role is to identify the various causes of accidents that occur in the battery production process, and to collaborate with Korean counterparts on translation tasks."""]
        )
        self.text1 = """다음은 공장 작업자가 녹음한 다양한 음석 파일을 기록한 파일입니다. 
당신에게는 3초 단위로 녹음된 3개의 Wave 세그먼트가 주어집니다.  또한 3초 단위로 동일한 프롬프트가 실행된다는 것을 명심하세요. 
즉, wav 파일 segment 가 다음과 같이 주어진다면, 
1,2,3,4,5,6,7
1,2,3
      3,4,5,
            5,6,7
형태로 마지막 wav segment는 호출할 때마다 중복되게 되니다. 
추가적으로 현재까지 Transcript된 내용 전체 원문과 함께 제공됩니다. 위에서 말한 중복되는 내용 말고, 새롭게 추가된 내용만을 output으로 전사(Transcription)하세요. 이는 output token 제한을 제어하기 위하여 필수적입니다. 
Confidence Score는 0 ~ 1 사이 부동소숫점 확률자로 얼마나 잘 전사되었는지 보여야 합니다.
Confidence Score가 높고, 이후 다음 번 전사 될 때 변경이 되지 않을 것 같다면, is_final에 "T"로 표시하고, 약간 애매해서 다음번 전사에서 변경될 소지가 높다면, is_final에 "F"로 표시해줘. 
이전 전사된 내용(previous_transcription)에서 is_final에 True로 표시된 문장/단어들은, 현재 진행하는 전사 결과에는 되도록 표시하지 말고 새로운 문장만 표기해 주세요. 
되도록이면 같은 Context로 된 긴 문장으로 표시해 주세요. 

<previous_transcription>
{previous_transcription}
</previous_transcription>

<audio_segments>"""
        self.text2 = """</audio_segments>
<output_example>
[
{{
  "seq_id" : ...,
  "timecode" : "00:00:00",
  "transcript_chinese" : ...,
  "translation_korean" : ...,
  "confidence_score" : ...,
  "is_final" :...,
}}, ...
]
</output_example>"""
        self.generation_config = {
            "max_output_tokens": 8192,
            "temperature": 0.4,
            "top_p": 0.95,
        }
        self.safety_settings = [
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=SafetySetting.HarmBlockThreshold.OFF
            ),
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=SafetySetting.HarmBlockThreshold.OFF
            ),
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=SafetySetting.HarmBlockThreshold.OFF
            ),
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=SafetySetting.HarmBlockThreshold.OFF
            ),
        ]
    
    def fix_json_quotes(self, json_string):
        return re.sub(r"(?<!\\)'", '"', json_string)

    def parse_gemini_response(self, json_str):
        start_index = json_str.find('```json') + 7
        end_index = json_str.find('```', start_index)
        json_str = json_str[start_index:end_index].strip()
        cleaned_string = json_str.replace('\\xa0', ' ')
        print(cleaned_string)
        try:
            return_json = json.loads(cleaned_string)
        except:
            return_json = json.loads(self.fix_json_quotes(cleaned_string))
        return return_json
    
    def call_gemini(self):
        previous_transcription = [item for sublist in self.last_3_transcripts for item in sublist]
        prompts = [self.text1.format(previous_transcription=previous_transcription)]
        for wav_data in self.last_3_wav_chunks:
            prompts.append(Part.from_data(data=wav_data,mime_type="audio/wav"))
        prompts.append(self.text2)
        response = self.gemini_model.generate_content(
          prompts,
          generation_config=self.generation_config,
          safety_settings=self.safety_settings,
          stream=False,
        )
        return self.parse_gemini_response(response.text)

    async def transcribe_stream(self, 
        audio_stream: AsyncGenerator[bytes, None],
        language_code: str
    ) -> AsyncGenerator[TranscriptionResult, None]:
        self.isProgressing = True
        async for chunk in audio_stream:
            if chunk:
                self.last_3_wav_chunks.append(chunk)
                if len(self.last_3_wav_chunks) > 3:
                    self.last_3_wav_chunks.pop(0)
                try:
                    new_transcriptions = self.call_gemini()
                    self.last_3_transcripts.append(new_transcriptions)
                    if len(self.last_3_transcripts) > 3:
                        self.last_3_transcripts.pop(0)
                    for transcript in new_transcriptions:
                        try:
                            new_result = TranscriptionResult(
                              transcript=transcript['transcript_chinese'],
                              is_final=transcript['is_final'],
                              language=language_code,
                              confidence=transcript['confidence_score'],
                              seq_id=transcript['seq_id'],
                              timecode=transcript['timecode'],
                              translation=transcript['translation_korean'],
                            )
                        except Exception as e:
                            print(e)
                            print("gemini result can't match the target object.")
                        yield new_result
                except Exception as e:
                    print(e)
                    print("gemini can't make right answer.")
            if not self.isProgressing:
                break
        print('exit normally')

    async def cleanup(self):
        print('cleanup called')
        if self.gemini_model:
            self.isProgressing = False
            #await self.client.close()
