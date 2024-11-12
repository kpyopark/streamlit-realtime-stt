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

    def transcription_to_testdata(self, original_text_json, transcription_text):
        outputs = []
        max_recursion = 10
        while True:
            max_recursion = max_recursion - 1
            if max_recursion < 0:
                break
            output = self.transcription_to_testdata_raw(original_text_json, transcription_text, outputs)
            outputs.extend(output["final"])
            if output["all_transcription_is_splitted_into_statement"]:
                break
        print(outputs)
        return {'final' : outputs}

    def transcription_to_testdata_raw(self, original_text_json, transcription_text, previous_output):
        prompt = f"""현재 Transcribe 작업을 수행 중에 있다. CER, WER, SER 평가를 위해서 ground truth가 되는 원문을 문장별로 구분한 자료가 "original_text_json"으로 제공한다.

주어진 <transcription_text> 자료는 Speech to Text실시간 스트리밍 방식으로 전사된 문장들이며, 문장구분이 되어 있지 않다. 
너는 주어진 <transcription_text> 스트링을 변형 하지 말고, 그냥 Split하는 위치를 잘 찾아서, "original_text_json"에 있는 seq_id별 문장과 제일 유사한 형태로 매칭하면된다.
output의 "transcription" 필드에는 위에서 Split한 문장으로 원문에 있는 seq_id에 해당하는 내용만 짤라서 넣어야 한다. 
Split한 문장이 너무 길다고 판단되면 다시 한번 재검토를 수행하면서 원 문장과 비교하면서 크기를 결정하여야 한다.
만약 transcription에 있는 문장과 너가 출력한 "final" 결과에 있는 문장이 다를 경우 10만 달러의 벌금을 매길 것이다.

먼저 Draft로 문장분리를 수행하고, 다시 원문과  Draft를 비교 검토하여, 빠진 seq_id가 없는지 문장 분리를 다시 한번 정확하게 수행한다. 
Draft본에서 seq_id가 없는 경우, 앞 seq_id에 있는 문장에 통합된 경우가 많다. 
예를 들어, draft version에 
"seq_id": 4, "statement_variation4 statement_variation5" 로 잘못 분리되었다면
최종 final output은
[ {{ "seq_id" : 4, "transcription" : "statement_variation4" , "original_text" : ... }}, {{ "seq_id" : 5, "transcription" : "statement_variation5" , "original_text" : ... }}, ... ]
형태로 "original_text"에 기반하여 최대한 추가 분리를 수행하여, 원본과 순서 및 텍스트 정합성을 맟추어야 한다.

Output Token이 제한되어 있어서 25문장이 넘어가면, 25문장까지만 답변을 해주면 되.
만약 <previous_output> 섹션에 값이 있다면 앞에서 구한 output 값이므로, previous_output안의 값에서 제일 마지막 값을 구하고, 현재 output은, 해당 seq_id 이후부터 다시 25문장까지 추가해서 답변을 해줘. 
예를 들어, <previous_output>에 있는 마지막 seq_id가 25라면, 그 다음문장부터 분리해서 26부터 시작하면 된다.
모든 문장을 분해하면, all_transcription_is_splitted_into_statement을 true로 설정하면 된다. 
예를 들어, original_text_json 마지막 seq_id가 23이라면, output의 seq_id도 23이면 마지막 문장이므로, all_transcription_is_splitted_into_statement을 true로 설정하면 된다.
자주, all_transcription_is_splitted_into_statement이 false로 설정되는 경우가 많아서, 원문 seq_id와 결과 출력물이 일치하는지 확인하면서, 최대한 정확하게 문장을 분리하면 된다.

<original_text_json>
{original_text_json}
</original_text_json>

<transcription_text>
{transcription_text}
</transcription_text>

<previous_output>
{previous_output}
</previous_output

<output example>
{{ 
"final" : [ {{
"seq_id" : ...,
"original_text" : ...,
"transcription": ...
}} ... ],
"all_transcription_is_splitted_into_statement" : ...
}}
</output example>

반드시 원문과 전사문장을 비교하면서, 문장을 분리하고, 원문과 비교하여, 문장이 맞는지 확인하고, 문장이 맞지 않는 경우, 다시 분리를 수행하면서, 원문과 비교하면서, 최대한 정확하게 분리를 수행해야 한다.

output json:"""
        
        model = self.get_model()
        response = model.generate_content(
            [prompt],
            generation_config=self.get_generation_config(),
            safety_settings=self.get_safety_settings(),
            stream=False,
        )
        print(prompt)
        full_response = self.parse_gemini_response(response.text)

        try:
            # Assuming the response is a properly formatted JSON string
            return full_response
        except json.JSONDecodeError:
            return None


    def split_original_text_to_statements(self, original_text):
        prompt = f"""현재 Transcribe 작업을 수행 중에 있다. CER, WER, SER 평가를 위해서 원문(original_text)을 문장별로 구분해서 json 형태로 출력해줘. 

<original_text>
{original_text}
</original_text>

<output example>
[ {{ \"seq_id\" : ...,
    \"original_text\" : ...,
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
