# -*- coding: utf-8 -*-
"""Ai.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vonf9wC7uK4NkEk-_WsZDYuEH4K2zDlv

---
# 01. 활경 설정
---
"""

import time

import os
import openai
import json
import unicodedata
import pandas as pd
from google.cloud import speech
from google.cloud import texttospeech
from datetime import datetime

# ChatGPT API Key Load
os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = os.environ["OPENAI_API_KEY"]

"""---
# 02. 사운드 입력 후 텍스트로 변환 (STT)
---
"""

#시작
start = time.time()

# Google Cloud 인증 키 파일 경로 (서비스 계정 키)
credentials_path = "C:\\Users\\win\\Desktop\\Ai\\ai-i-401313-176ecd5ad2cf.json"
# Google Cloud 클라이언트 초기화
client = speech.SpeechClient.from_service_account_json(credentials_path)

# 오디오 파일을 읽어옵니다.
audio_file_path = "C:\\Users\\win\\Desktop\\Ai\\input.mp3"
with open(audio_file_path, 'rb') as audio_file:
    content = audio_file.read()

audio = speech.RecognitionAudio(content=content)

# 음성 인식 요청 생성
config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED,
    sample_rate_hertz=16000,  # 오디오 샘플 속도에 따라 조정
    language_code='ko-KR'  # 인식할 언어 코드 지정
)

# 음성 인식 요청 보내기
response = client.recognize(config=config, audio=audio)

# Get the transcript from the response
transcript_text = response.results[0].alternatives[0].transcript if response.results else ""

# 유니코드 문자열을 한국어 문자열로 변환
Message_text = unicodedata.normalize("NFC", transcript_text)

# 줄 바꿈 추가
Message_text = Message_text.replace('. ', '.\n').replace('? ', '?\n').replace('! ', '!\n')

# 출력 결과
print("STT 결과")
print(Message_text)
with open("C:\\Users\\win\\Desktop\\Ai\\example.txt", "w") as file:
    file.write(Message_text)

#끝
end = time.time()
print(f"STT Time: {end-start:.5f}sec")
print("===========================================================================================")
Message_text = "배워도 배워도 끝이 없더라고요 자신이 없다 보니까 여전히 찾기 위해서 계속 방황을 하고 있는 중인 것 같습니다"

"""---
# 03. Chat GPT를 통한 요약
---
"""

#시작
start = time.time()
# 모델 - GPT 3.5 Turbo 선택
model = "gpt-4"

# 질문 작성하기
#query = "다음 문서를 요약해줘: " + Message_text
query =f"""
진심의 중요성: 위로의 내용보다는 그 내용을 전달하는 마음, 즉 '진심'이 더 중요하다. 진심이 담긴 위로는 상대방에게 효과적으로 전달된다.

위로의 세 가지 원칙:

명료화 (Validation): 상대방의 감정을 읽어주고 인정해주는 것. 예: "마음이 아팠겠다."
정상화 (Normalizing): 상대방의 감정 반응이 그 상황에서는 당연하다는 것을 알려주는 것. 예: "그런 상황이라면 화가 나는 게 당연해."
승인 (Affirmation): 상대방의 진정한 가치를 인정해주는 것. 예: "너는 소중한 존재다."
힐링의 중요성: 사람들이 위로를 받고 싶어하는 이유는 마음이 따뜻해지면 사람이 움직이게 되고, 치유하는 방향으로 나아가게 되기 때문이다.

상기 글을 참고하세요. 그 후, 다음 글을 친구의 입장이 되어서 간단 명료하게 답하시오.
"{Message_text}"
"""
# 메시지 설정하기
messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": query}
]

# ChatGPT API 호출하기
response = openai.ChatCompletion.create(
    model=model,
    messages=messages
)
answer = response['choices'][0]['message']['content']
print("ChatGPT: "+ answer.replace('.', '.\n').replace('? ', '?\n').replace('! ', '!\n'))
#끝
end = time.time()
print(f"GPT Completion Time: {end-start:.5f}sec")
print("===========================================================================================")


# # 모델 - GPT 3.5 Turbo 선택
# model = "gpt-3.5-turbo"

# # 질문 작성하기
# #query = "다음 문서를 요약해줘: " + Message_text
# query = Message_text
# # 메시지 설정하기
# messages = [
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": query}
# ]

# # ChatGPT API 호출하기
# response = openai.ChatCompletion.create(
#     model=model,
#     messages=messages
# )
# answer = response['choices'][0]['message']['content']
# print("ChatGPT: "+ answer.replace('.', '.\n').replace('? ', '?\n').replace('! ', '!\n'))

"""---
# 04. 텍스트를 사운드로 변환 (TTS)
---
"""

#시작
start = time.time()
# Google Cloud TTS 인증 키 파일 경로 (서비스 계정 키)
tts_credentials_path = "C:\\Users\\win\\Desktop\\Ai\\ai-i-401313-176ecd5ad2cf.json"

# TTS 클라이언트 초기화
tts_client = texttospeech.TextToSpeechClient.from_service_account_json(tts_credentials_path)


def synthesize_text_to_audio(text, output_filename="C:\\Users\\win\\Desktop\\Ai\\tts_output.mp3"):
    # 텍스트 설정
    synthesis_input = texttospeech.SynthesisInput(text=text)

    # 음성 설정
    voice = texttospeech.VoiceSelectionParams(
        name="ko-KR-Standard-D",
        language_code="ko-KR",
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
    )

    # 오디오 설정
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    # TTS 요청 및 응답
    response = tts_client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    # 오디오 파일로 저장
    with open(output_filename, "wb") as out:
        out.write(response.audio_content)
        print(f"Audio content written to '{output_filename}'")
        
# 현재 시간을 얻습니다.
current_time = datetime.now()
# "년월일시분" 형식으로 포맷팅합니다.
formatted_time = current_time.strftime("%Y%m%d%H%M%S")
# ChatGPT로부터 얻은 답변을 음성으로 변환
synthesize_text_to_audio(answer, output_filename = "C:\\Users\\win\\Desktop\\Ai\\output_audio\\" + formatted_time + ".mp3")
#끝
end = time.time()
print(f"TTS Time: {end-start:.5f}sec")
print("===========================================================================================")

"""Program Time total  
- gpt-3.5-turbo  
STT(1.35) + GPT(33.26) + TTS( 0.56) ≈ 35(sec)

- gpt-4
STT(1.35) + GPT(10.23) + TTS( 0.56) ≈ 12(sec)
"""










