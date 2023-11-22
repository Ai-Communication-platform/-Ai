"""Ai.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vonf9wC7uK4NkEk-_WsZDYuEH4K2zDlv

---
# 01. 활경 설정
---
"""
import schedule
import time
import os
import openai
import json
import unicodedata
import pandas as pd
from google.cloud import speech
from google.cloud import texttospeech
from datetime import datetime, timedelta
import pygame
import firebase_admin
from firebase_admin import credentials, initialize_app, storage
import requests

# 서비스 계정 키(JSON 파일)의 경로
cred = credentials.Certificate("C:\\Users\\win\\Documents\\ai-firebase-f501e-firebase-adminsdk-pgie0-832a8c2eb2.json")
# 로컬에 저장할 디렉토리 지정 (예: '/your/local/directory/')
path = 'C:\\Users\\win\\Documents\\-Ai\\web\\js\\uploads'
# Google Cloud 인증 키 파일 경로 (서비스 계정 키)
credentials_path = "C:\\Users\\win\\Documents\\ai-i-401313-92d1dd2e0014.json"

# ChatGPT API Key Load
os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = os.environ["OPENAI_API_KEY"]

# Google Cloud TTS 인증 키 파일 경로 (서비스 계정 키)
tts_credentials_path = "C:\\Users\\win\\Documents\\ai-i-401313-92d1dd2e0014.json"
# 마지막으로 확인한 시간을 기록하는 전역 변수
last_checked_time = time.time()

# Firebase Admin SDK 초기화
firebase_admin.initialize_app(cred, {
    'storageBucket': "ai-firebase-f501e.appspot.com"
})


"""---
# 02. 사운드 입력 후 텍스트로 변환 (STT)
---
"""

def job():  
    global last_checked_time
    global path

    start = time.time()
    # Storage 버킷 접근
    bucket = storage.bucket()

    # Storage 내의 MP3 파일 목록 가져오기
    blobs = bucket.list_blobs()
    mp3_files = [blob for blob in blobs if blob.name.endswith('.mp3')]

    # 파일의 생성 날짜를 기준으로 최신 파일 찾기
    latest_file = max(mp3_files, key=lambda x: x.time_created)

    # 파일 다운로드 URL 가져오기
    download_url = latest_file.generate_signed_url(timedelta(seconds=300), method='GET')


    if not os.path.exists(path):
        os.makedirs(path)
        
    # 파일의 원래 이름을 유지하여 로컬 디렉토리에 저장
    local_file_path = os.path.join(path, latest_file.name.replace('/',''))
    response = requests.get(download_url)
    with open(local_file_path, 'wb') as file:
        file.write(response.content)
        time.sleep(0.5)
    end = time.time()
    print(f"Storage Time: {end-start:.5f}sec")

    # 현재 시간과 마지막으로 확인한 시간 사이에 생성된 모든 파일을 찾습니다.
    for file_name in os.listdir(path):
        if file_name.endswith('.mp3'):
            file_path = os.path.join(path, file_name)
            file_creation_time = os.path.getctime(file_path)

            # print(f"File Creation Time: {file_creation_time}, Last Checked Time: {last_checked_time}")  # 시간 비교 출력

            # 파일 생성 시간이 마지막으로 확인한 시간보다 이후인 경우
            if file_creation_time > last_checked_time:
                print("들어옴")
                # print(f'New file detected: {file_name}')
                # 여기에 새 파일이 발견될 때 실행할 코드를 추가합니다.
                # Google Cloud 클라이언트 초기화
                client = speech.SpeechClient.from_service_account_json(credentials_path)
    
                # print(file_path)
                # # Initialize pygame
                # pygame.init()
                # # Load the MP3 file
                # pygame.mixer.music.load(rf"{file_path}")
                # # Play the music
                # pygame.mixer.music.play()


                # stt시자
                # 오디오 파일을 읽어옵니다.
                audio_file_path = os.path.join(path, file_name)
                with open(audio_file_path, 'rb') as audio_file:
                    content = audio_file.read()
                start = time.time()
                audio = speech.RecognitionAudio(content=content)
                # 음성 인식 요청 생성
                config = speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.MP3,
                    sample_rate_hertz=16000,  # 오디오 샘플 속도에 따라 조정
                    language_code='ko-KR'  # 인식할 언어 코드 지정
                )
                # 음성 인식 요청 보내기
                response = client.recognize(config=config, audio=audio)
                # Get the transcript from the response
                transcript_text = response.results[0].alternatives[0].transcript if response.results else "stt안됨"
                # 유니코드 문자열을 한국어 문자열로 변환
                Message_text = unicodedata.normalize("NFC", transcript_text)
                # 줄 바꿈 추가
                Message_text = Message_text.replace('. ', '.\n').replace('? ', '?\n').replace('! ', '!\n')
                # 출력 결과
                print("STT 결과")
                print(Message_text)
                with open("C:\\Users\\win\\Documents\\-Ai\\example.txt", "w") as file:
                    file.write(Message_text)
                #끝
                end = time.time()
                print(f"STT Time: {end-start:.5f}sec")
                print("===========================================================================================")
                # Message_text = "배워도 배워도 끝이 없더라고요 자신이 없다 보니까 여전히 찾기 위해서 계속 방황을 하고 있는 중인 것 같습니다"
                """---
                # 03. Chat GPT를 통한 요약
                ---
                """

                #시작
                start = time.time()
                # 모델 - GPT 3.5 Turbo 선택
                model = "gpt-3.5-turbo-1106"
                print("model: ", model)
                # 질문 작성하기
                #query = "다음 문서를 요약해줘: " + Message_text
                query =f"""
                Importance of sincerity: Rather than the content of comfort, the heart conveying the content, or ‘sincerity,’ is more important. Sincere consolation is effectively conveyed to the other person.

                Three principles of comfort:

                Validation: Reading and acknowledging the other person’s feelings. Example: “I must have been heartbroken.”
                Normalizing: Letting the other person know that their emotional response is natural in that situation. Example: “It’s natural to feel angry in that situation.”
                Affirmation: Recognizing the true value of the other person. Example: “You are precious.”
                Importance of Healing: The reason people want to be comforted is because when their hearts are warm, people move and move in the direction of healing.

                Please refer to the above article. After that, answer the following passage simply and clearly, as if you were from the perspective of a friend of the same age. Only answers are output. Answer in Korean.                
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

                #TTS시작
                start = time.time()
    
                # TTS 클라이언트 초기화
                tts_client = texttospeech.TextToSpeechClient.from_service_account_json(tts_credentials_path)
                def synthesize_text_to_audio(text, output_filename="C:\\Users\\ewqds\\Documents\\GitHub\\-Ai\\tts_output.mp3"):
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
                
                # 현재 시간
                current_time = datetime.now()
                # "년월일시분" 형식으로 포맷팅합니다.
                formatted_time = current_time.strftime("%Y%m%d%H%M%S")
                # ChatGPT로부터 얻은 답변을 음성으로 변환
                synthesize_text_to_audio(answer, output_filename = "C:\\Users\\win\\Documents\\-Ai\\output_audio\\" + formatted_time + ".mp3")
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
                # 마지막으로 확인한 시간을 업데이트합니다.
                # print("시간 :", last_checked_time)
                last_checked_time = time.time()

                # Initialize pygame
                pygame.init()
                # Load the MP3 file
                pygame.mixer.music.load("C:\\Users\\win\\Documents\\-Ai\\output_audio\\" + formatted_time + ".mp3")
                # Play the music
                pygame.mixer.music.play()

                # Wait for the music to play completely
                while pygame.mixer.music.get_busy():
                    time.sleep(1)

# 매 분마다 job 함수를 실행합니다.
schedule.every(5).seconds.do(job)

if __name__ == "__main__":
    while True:
        schedule.run_pending()
        time.sleep(1)
