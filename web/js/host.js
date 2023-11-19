const express = require('express');
const multer = require('multer');
const ffmpeg = require('fluent-ffmpeg');
const fs = require('fs');

const app = express();

// 정적 파일을 제공하는 폴더 설정
app.use(express.static('C:\\Users\\ewqds\\Documents\\GitHub\\-Ai\\web'));
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