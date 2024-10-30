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
        self.last_n = 1
        self.last_n_transcripts = []
        self.last_n_wav_chunks = []

    async def initialize(self):
        vertexai.init(project=self.project_id, location=self.location)
        self.gemini_model = GenerativeModel(
          "gemini-1.5-flash-002",
          #system_instruction=["""You are a simultaneous interpreter working at a Chinese battery manufacturing plant for electric vehicles. Your role is to identify the various causes of accidents that occur in the battery production process, and to collaborate with Korean counterparts on translation tasks."""]
        )
        self.text1 = """You are a high-performance speech-to-text transcription system. Your primary goal is to avoid repetition while maintaining forward progression in transcription.
INPUT COMPONENTS:

<audio_segments>: 4-second audio chunks with 0.5s overlap
<previous_transcription>: Previously transcribed content
<terms>: Reference terminology list

CORE PRINCIPLE:
Never repeat content from <previous_transcription> - always move forward, even if uncertain about current segment.
TRANSCRIPTION RULES:

Forward-Only Processing:

Process only new content from each audio segment
If overlap contains previously transcribed content, skip it entirely
Focus only on transcribing content that appears after the last word in <previous_transcription>
Prioritize continuity over perfection


Overlap Handling:

When detecting overlapped content with previous transcription:

SKIP all content until finding new speech
DO NOT attempt to correct previous transcriptions unless critically wrong
If uncertain, prefer omission over repetition




Error Tolerance:

Accept potential errors rather than revisiting previous content
Only mark revisions if they are critically important (e.g., completely wrong meaning)
Set lower confidence scores for uncertain segments but continue forward


Term Reference:

Use <terms> list for reference but don't backtrack to correct previous uses
Apply terms knowledge only to new content

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
</output_example>

<terms>
<term1>
<bad pronounciation> home error </bad pronounciation>
<correct pronounciation> horn anvil </correct pronounciation>
</term1>
</terms>

STRICT RULES:

DO NOT output any segment if its content is already in <previous_transcription>
DO NOT attempt to fix minor errors in previous transcriptions
DO NOT reprocess any content that was already transcribed
DO NOT make uncertain corrections to previous transcriptions and responses
Always move forward in the audio, even if uncertain
Set is_final to false only for segments that are critically uncertain

"""
        self.generation_config = {
            "max_output_tokens": 8192,
            "temperature": 0.3,
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
        previous_transcription = [item for sublist in self.last_n_transcripts for item in sublist]
        prompts = [self.text1.format(previous_transcription=previous_transcription)]
        for wav_data in self.last_n_wav_chunks:
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
                self.last_n_wav_chunks.append(chunk)
                if len(self.last_n_wav_chunks) > self.last_n:
                    self.last_n_wav_chunks.pop(0)
                try:
                    new_transcriptions = self.call_gemini()
                    self.last_n_transcripts.append(new_transcriptions)
                    if len(self.last_n_transcripts) > self.last_n:
                        self.last_n_transcripts.pop(0)
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
