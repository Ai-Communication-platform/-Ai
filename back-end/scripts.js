

// // 자동 스크롤 기능
// const messageContainer = document.getElementById('message-container');
// const scrollButton = document.getElementById('scroll-button');
// scrollButton.addEventListener('click', () => {
//   messageContainer.scrollTop = messageContainer.scrollHeight;
// });
const admin = require('firebase-admin');
const axios = require('axios');
const fs = require('fs');

// Firebase 초기화
admin.initializeApp({
    credential: admin.credential.cert(serviceAccount),
    storageBucket: 'gs://ai-i-401313.appspot.com'
  });

const bucket = admin.storage().bucket();

async function downloadLatestFile(directory, destination) {
  // 지정된 디렉토리에서 파일 목록 조회
  const [files] = await bucket.getFiles({ prefix: directory });

  // 파일을 생성 시간별로 정렬
  const sortedFiles = files.sort((a, b) => b.metadata.timeCreated.localeCompare(a.metadata.timeCreated));

  // 가장 최신 파일 찾기
  const latestFile = sortedFiles[0];

  // 다운로드 URL 생성
  const [url] = await latestFile.getSignedUrl({
    action: 'read',
    expires: '03-17-2025'
  });

  // 파일 다운로드
  const response = await axios({
    url,
    method: 'GET',
    responseType: 'stream'
  });

  return new Promise((resolve, reject) => {
    const writer = fs.createWriteStream(destination);
    response.data.pipe(writer);

    writer.on('finish', resolve);
    writer.on('error', reject);
  });
}

//녹음 파일명을 현재 시간으로 저장
function getCurrentTimestamp() {
    const now = new Date();

    const year = now.getFullYear();
    const month = String(now.getMonth() + 1).padStart(2, '0'); // 월은 0부터 시작하므로 1을 더합니다
    const day = String(now.getDate()).padStart(2, '0');
    const hours = String(now.getHours()).padStart(2, '0');
    const minutes = String(now.getMinutes()).padStart(2, '0');
    const seconds = String(now.getSeconds()).padStart(2, '0');

    return `${year}${month}${day}${hours}${minutes}${seconds}`;
}

// 파일명 생성
const fileName = `${getCurrentTimestamp()}.mp3`;

// '녹음 파일이 저장된 디렉토리 경로'와 '로컬 저장 경로' 지정
downloadLatestFile('gs://ai-i-401313.appspot.com', 'C:\\Users\\win\\Documents\\GitHub\\-Ai\\back-end\\uploads\\'+getCurrentTimestamp)
  .then(() => console.log('Latest file downloaded successfully.'))
  .catch(err => console.error('Error downloading file:', err));





const recordButton = document.getElementById('record-button');

// 클라이언트 측 JavaScript
let mediaRecorder;
let audioChunks = [];

recordButton.addEventListener('click', () => {
    if (mediaRecorder && mediaRecorder.state === "recording") {
        mediaRecorder.stop();
    } else {
        navigator.mediaDevices.getUserMedia({ audio: true })
            .then(stream => {
                mediaRecorder = new MediaRecorder(stream);
                mediaRecorder.ondataavailable = event => {
                    audioChunks.push(event.data);
                };
                mediaRecorder.onstop = async () => {
                    // 녹음된 오디오 데이터를 Blob으로 변환
                    const audioBlob = new Blob(audioChunks, { 'type' : 'audio/ogg; codecs=opus' });
                    const formData = new FormData();
                    formData.append('audioFile', audioBlob);

                    // 서버로 데이터 전송
                    try {
                        const response = await fetch('http://localhost:3000/upload', {
                            method: 'POST',
                            body: formData
                        });
                        if (!response.ok) {
                            throw new Error(`Server responded with ${response.status}`);
                        }
                        const result = await response.json();
                        console.log(result.message);
                    } catch (error) {
                        console.error('Upload error:', error);
                    }
                };
                mediaRecorder.start();
                audioChunks = [];
            })
            .catch(e => console.error('getUserMedia() error:', e));
    }
});