const express = require('express');
const cors = require('cors'); // cors 모듈 요구

const multer = require('multer');
const fs = require('fs');
const ffmpeg = require('fluent-ffmpeg');
// ffmpeg 실행 파일의 경로를 설정
ffmpeg.setFfmpegPath('C:\\Users\\ewqds\\Documents\\GitHub\\-Ai\\web\\js\\node_modules\\ffmpeg-6.1-essentials_build\\bin\\ffmpeg.exe');

const app = express();
app.use(cors()); // CORS 미들웨어 적용

const upload = multer({ dest: 'uploads/' });


app.post('/upload', upload.single('audioFile'), (req, res) => {
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