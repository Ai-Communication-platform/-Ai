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
import numpy as np
from datetime import datetime, timedelta
import requests
from moviepy.editor import AudioFileClip

def chatgpt_call(model, messages):
    # ChatGPT API 호출하기
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages
    )
    return response

# ChatGPT API Key Load
os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = os.environ["OPENAI_API_KEY"]


# 마지막으로 확인한 시간을 기록하는 전역 변수
last_checked_time = time.time()

# 감정분석을 위한 프롬프트 읽어오기
generation_prompt = open('C:\\Users\\win\\Documents\\GitHub\\-Ai\\prompt\\generation_Ai.txt', "r", encoding='utf-8').read()
#print(generation_prompt)

# 모델 - GPT 3.5 Turbo 선택
model = "gpt-4"
print("model: ", model)
#==================#
#   감정 분석하기   |
#==================#

# 사용자의 현재 감정 상태와 상황이 요약되어 Message에 합쳐짐.
data = pd.read_csv('C:\\Users\\win\\Documents\\GitHub\\-Ai\\감성대화말뭉치(최종데이터)_Training.csv')
test_x, test_y = np.array(data['사람문장1']), np.array(data[['감정_대분류', '감정_소분류']])
output_file = 'C:\\Users\\win\\Documents\\GitHub\\-Ai\\sentence.txt'
with open(output_file, 'w', encoding='utf-8') as file:
    for i, value in enumerate(test_x):
        line = f"{i+1}: {value}\n"
        file.write(line)

data_size = 100
pred = []
# print(test_x)
# print(test_y)
# print(len(test_y))
for size in range(10, 110, 10):
    test_x, test_y = test_x[:size], test_y[:size]
    for sample, label in zip(test_x, test_y): 
        # print("sample: ", sample)
        # print("label: ", label)
        prompt = generation_prompt.format(Document=sample)
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
        element = list(Sammary.split(' '))
        print(element)
        pred.append([element[2], element[5]])
        print(f"sentiment analysis Time: {end-start:.5f}sec")
        time.sleep(20)
print("감정 예측 결과: ")
print(pred)
#==================#

error = 0
acc = 0
for predict, target in zip(pred, label):
    if predict != target:
        # 하나만 맞는 경우에는 error 0.5점
        if predict[0] == target[0] or predict[1] == target[1]:
            error += 0.5
        # 다 틀렸으면 error 1점
        else:
            error += 1

acc = (data_size - error)/data_size

print("accuracy: ", acc)
