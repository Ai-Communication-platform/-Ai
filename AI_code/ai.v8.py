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
from moviepy.editor import AudioFileClip

# 서비스 계정 키(JSON 파일)의 경로
cred = credentials.Certificate("C:\\Users\\win\\Documents\\ai-firebase-f501e-firebase-adminsdk-pgie0-832a8c2eb2.json")
# 로컬에 저장할 디렉토리 지정 (예: '/your/local/directory/')
path = 'C:\\Users\\win\\Documents\\GitHub\\-Ai\\input_audio'
# Google Cloud 인증 키 파일 경로 (서비스 계정 키)
credentials_path = "C:\\Users\\win\\Documents\\ai-i-401313-176ecd5ad2cf.json"

# ChatGPT API Key Load
os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = os.environ["OPENAI_API_KEY"]

# Google Cloud TTS 인증 키 파일 경로 (서비스 계정 키)
tts_credentials_path = "C:\\Users\\win\\Documents\\ai-i-401313-176ecd5ad2cf.json"
# 마지막으로 확인한 시간을 기록하는 전역 변수
last_checked_time = time.time()

# Firebase Admin SDK 초기화
bucket_name = 'ai-firebase-f501e.appspot.com'
firebase_admin.initialize_app(cred, {
    'storageBucket': 'ai-firebase-f501e.appspot.com'
})

# 감정분석을 위한 프롬프트 읽어오기
generation_prompt = open('C:\\Users\\win\\Documents\\GitHub\\-Ai\\prompt\\generation_Ai.txt', "r", encoding='utf-8').read()
#print(generation_prompt)


def chatgpt_call(model, messages):
    # ChatGPT API 호출하기
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages
    )

    return response

def job():  
    global last_checked_time
    global path

    start = time.time()
    # Storage 버킷 접근
    bucket = storage.bucket()
    
    # Storage 내의 MP3 파일 목록 가져오기
    blobs = bucket.list_blobs(prefix="files/")
    mp3_files = [blob for blob in blobs if blob.name.endswith('.mp3')]
    # 파일의 생성 날짜를 기준으로 최신 파일 찾기
    if len(mp3_files) == 0:
        return

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
                # print("사용자 입력 들어옴")
                # print(f'New file detected: {file_name}')
                # 여기에 새 파일이 발견될 때 실행할 코드를 추가합니다.
                # Google Cloud 클라이언트 초기화
                client = speech.SpeechClient.from_service_account_json(credentials_path)
    
                # Convert the AAC-encoded file to MP3 format using moviepy
                
                audio_clip = AudioFileClip(file_path)
                audio_clip.write_audiofile(file_path, codec='mp3')

                # print(file_path)
                # # Initialize pygame
                # pygame.init()
                # # Load the MP3 file
                # pygame.mixer.music.load(rf"{file_path}")
                # # Play the music
                # pygame.mixer.music.play()


                # stt시자
                # 오디오 파일을 읽어옵니다.
                with open(file_path, 'rb') as audio_file:
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
                with open("C:\\Users\\win\\Documents\\GitHub\\-Ai\\example.txt", "w") as file:
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


                # 모델 - GPT 3.5 Turbo 선택
                model = "gpt-3.5-turbo-1106"
                print("model: ", model)

                #==================#
            #   |   감정 분석하기   |
                #==================#
                # 사용자의 현재 감정 상태와 상황이 요약되어 Message에 합쳐짐.
                prompt = generation_prompt.format(Document=Message_text)
                # 메시지 설정하기
                messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                ]
                start = time.time()
                # 감정 분석 chatgpt로 진행
                Sammary = chatgpt_call(model, messages)['choices'][0]['message']['content']
                end = time.time()
                print("감정 분석 결과: ")
                print(Sammary)
                print(f"sentiment analysis Time: {end-start:.5f}sec")
                #==================#


                # 질문 작성하기
                #query = "다음 문서를 요약해줘: " + Message_text
                query =f"""
                [상대방의 입장을 충분히 이해하고 공감하는 대화방식]
                 1. 상대방의 감정 이해하기: 상대방이 자신의 감정을 표현할 때 혼자가 아니라는 느낌을 갖게 하는 것이 중요합니다. 
                    이는 연결감을 형성하고 공감의 진정한 힘을 활용하는 데 도움이 됩니다.

                 2. 자신의 생각과 판단을 내려놓기: 다른 사람을 이해하려면 자신의 생각, 가치 평가, 판단 기준을 일시적으로 내려놓아야 합니다. 
                    이를 통해 상대방은 자신이 공감받고 있다는 느낌을 받을 수 있다.

                 3. 공감과 공감의 차이 인식: 공감은 상대방과 같은 감정을 느끼는 것이 아니라, 상대방의 입장에서 이야기를 듣고, 상대방의 감정을 이해하는 것입니다. 
                    여기에는 다른 사람의 입장에서 서서 그들의 감정적 여정을 따라가는 것이 포함됩니다.

                 4. 열린 마음으로 듣기: 공감은 말로만 시작되는 것이 아니라 마음을 비우고 상대방의 말에 집중하는 것에서부터 시작됩니다. 
                    상대방의 상황을 이해하고, 그들의 입장에서 대화에 참여하는 것이 중요합니다.

                 이러한 방법을 사용하면 다른 사람의 입장을 더 잘 이해하고 대화에서 진정한 공감을 나눌 수 있습니다. 
                 다른 사람의 감정과 경험을 존중하고, 그 사람의 입장에서 생각하고, 진심으로 듣는 것이 중요합니다.

                 [진심의 중요성]
                 1. 검증: 상대방의 감정을 읽고 인정합니다. 예: “당신은 상심한 것 같습니다.”
                 정상화: 상황에 대한 감정적 반응이 자연스러운 것임을 상대방에게 알리는 것입니다. 예: “그런 상황에서 화가 나는 것은 정상입니다.”
                 2. 긍정: 상대방의 진정한 가치를 인정합니다. 예: “당신은 소중한 사람입니다.”
                 3. 치유의 중요성: 사람들은 마음에 따뜻함을 느낄 때 치유를 향해 나아가는 경향이 있기 때문에 위로를 추구합니다.

                 편안함의 내용보다 그것을 전달하는 '진심성'이 중요합니다. 진심 어린 위로가 상대방에게 효과적으로 전달됩니다.

                 사용자의 현재 감정과 상황:
                "{Sammary}

                위의 글과 이용자의 현재 상황 및 감정을 참고하시기 바랍니다. 그 후, 
                동갑내기 친구의 입장에서 생각하듯이 다음 문장들에 간단명료하게 답해 보세요. 
                답변만 인쇄됩니다. 
                한국어로 답변해주세요.
                "{Message_text}"
                """
                # 메시지 설정하기
                messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": query}
                ]

                # ChatGPT API 호출하기
                # GPTCompletion 시작
                start = time.time()
                response = chatgpt_call(model, messages)

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
                formatted_time = current_time.strftime("%Y-%m-%d-%H-%M-%S")

                # ChatGPT로부터 얻은 답변을 음성으로 변환
                output_filename = "C:\\Users\\win\\Documents\\GitHub\\-Ai\\output_audio\\" + formatted_time + ".mp3"
                
                # ACC 포매팅에서 저장
                synthesize_text_to_audio(answer, output_filename = output_filename)
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

                # 사용 예
                start = time.time()
                destination_blob_name = 'output/' + formatted_time + ".mp3"

                upload_blob = bucket.blob(destination_blob_name)
                with open(output_filename, 'rb') as file:
                    upload_blob.upload_from_file(file)
                end = time.time()
                print(f"data send Time: {end-start:.5f}sec")
    
                # # Initialize pygame
                # pygame.init()
                # # Load the MP3 file
                # pygame.mixer.music.load("C:\\Users\\win\\Documents\\GitHub\\-Ai\\output_audio\\" + formatted_time + ".mp3")
                # # Play the music
                # pygame.mixer.music.play()

                # Wait for the music to play completely
                while pygame.mixer.music.get_busy():
                    time.sleep(1)
                # 마지막으로 확인한 시간을 업데이트합니다.
                # print("시간 :", last_checked_time)
                last_checked_time = time.time()


# 매 분마다 job 함수를 실행합니다.
global_start = time.time()
schedule.every(5).seconds.do(job)
global_end   = time.time()
print(f"All Time: {global_end-global_start:.5f}sec")

if __name__ == "__main__":
    while True:
        schedule.run_pending()
        time.sleep(1)
