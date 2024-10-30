# llm_utility.py
import vertexai
from vertexai.generative_models import GenerativeModel, Part, SafetySetting
import json

class GeminiAPI:
    def __init__(self, project_id, location):
        self.project_id = project_id
        self.location = location
        self.init_vertex_ai()
        
    def init_vertex_ai(self):
        vertexai.init(project=self.project_id, location=self.location)
        
    def get_model(self):
        system_instruction = """당신은 LG Energy Solution에서 근무하는 중국 현지 통역 담당자입니다. 
        당신의 역할은 현지 채용 근로자들에 대한 번역 및 전사(Transcription)작업 및 동시 통역 작업을 주 업무로 하고 있습니다. 
        LG Energy Solution은 2차 전지 관련 글로벌 회사입니다."""
        
        return GenerativeModel(
            "gemini-1.5-flash-002",
            system_instruction=[system_instruction]
        )
        
    def get_safety_settings(self):
        return [
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
    
    def get_generation_config(self):
        return {
            "max_output_tokens": 8192,
            "temperature": 1,
            "top_p": 0.95,
        }
    
    def parse_gemini_response(self, json_str):
        start_index = json_str.find('```json') + 7
        end_index = json_str.find('```', start_index)
        json_str = json_str[start_index:end_index].strip()
        cleaned_string = json_str.replace('\\xa0', ' ')
        print(cleaned_string)
        try:
            return_json = json.loads(cleaned_string)
        except:
            return_json = json.loads(fix_json_quotes(cleaned_string))
        return return_json
        
    def transcription_to_testdata(self, transcription_text):
        prompt = f"""다음 주어진 전사 원문을 이용하여 향후, Transcribe 모델(Speech to Text)에 대한 시험평가를 준비하고 있다. 
        해당 원문에 대해서 문장 별로 분리해서 표시해줘. 

        문장 분리를 진행할 때, 결과 인자 하나씩을 문장으로 인식하고 분리해주세요. 개별 문장에는 하나의 인자만 포함되어야 합니다. 불필요한 앞에서 언급한 원인을 추가하지 마세요. Transcribe 모델이기 때문에 원문을 그대로 유지하는 것이 제일 중요합니다.
        또한, 사람이 읽지 못하는 캐릭터는 제거하고 그 자리를 공백으로 치환해 주세요. **예를 들어 ")" ":" 이런 캐릭터는 반드시 반드시 제외해야 합니다. **

        예시) 원인 : 공장 파이프라인 중단 -> 원인 공장 파이프라인 중단
        
        결과를 생각하고 다시한번 : , ":", "," 와 같이 사람이 읽지 못하는 문자가 들어가 있는지 확인해 보고, 이후에 결과를 적어주세요.

        <전사원문>
        {transcription_text}
        </전사원문>

        <output example>
        [ {{ \"seq_id\" : ...,
            \"text\" : ...
        }}, ... ]
        </output example>"""
        
        model = self.get_model()
        response = model.generate_content(
            [prompt],
            generation_config=self.get_generation_config(),
            safety_settings=self.get_safety_settings(),
            stream=False,
        )

        full_response = self.parse_gemini_response(response.text)

        try:
            # Assuming the response is a properly formatted JSON string
            return full_response
        except json.JSONDecodeError:
            return None

