const express = require('express');
const cors = require('cors'); // cors 모듈 요구
const multer = require('multer');
const ffmpeg = require('fluent-ffmpeg');
const admin = require('firebase-admin');
const axios = require('axios');
const fs = require('fs');




// Firebase 초기화
admin.initializeApp({
    credential: admin.credential.cert('C:\\Users\\win\\Documents\\google-services.json'),
    storageBucket: 'gs://ai-firebase-f501e.appspot.com/files',
    project_id: 'ai-firebase-f501e'
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



// ffmpeg 실행 파일의 경로를 설정
ffmpeg.setFfmpegPath('C:\\Users\\ewqds\\Documents\\GitHub\\-Ai\\back-end\\node_modules\\ffmpeg-6.1-essentials_build\\bin\\ffmpeg.exe');

const app = express();
app.use(cors()); // CORS 미들웨어 적용

const upload = multer({ dest: 'uploads/' });


app.post('/upload', upload.single('audioFile'), (req, res) => {
    //encoding할 파일 경로
    const filePath = req.file.path;
    const outputFilePath = req.file.path + '.mp3';

    ffmpeg(filePath)
        .toFormat('mp3')
        .on('end', function() {
            console.log('File has been converted successfully');
            fs.unlink(filePath, (err) => { // 원본 파일 삭제
                if (err) console.error('Error deleting original file:', err);
            });
            res.send({ message: 'Audio converted to MP3', file: outputFilePath });
        })
        .on('error', function(err) {
            console.error('Error converting file:', err);
            res.status(500).send('Error converting audio');
        })
        .save(outputFilePath);
});


const port = 3000;
app.listen(port, () => {
    console.log(`Server running on http://localhost:${port}`);
});