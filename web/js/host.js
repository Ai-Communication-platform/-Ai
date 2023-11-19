const express = require('express');
const multer = require('multer');
const lamejs = require('lamejs');
const fs = require('fs');

const app = express();

// 정적 파일을 제공하는 폴더 설정
app.use(express.static('public'));
const upload = multer({ dest: 'uploads/' });

app.post('/upload', upload.single('audioFile'), (req, res) => {
    const filePath = req.file.path;
    const mp3FilePath = filePath.replace('.wav', '.mp3');

    fs.readFile(filePath, (err, buffer) => {
        if (err) {
            console.error(err);
            return res.status(500).send('Error reading file');
        }

        // WAV 파일을 MP3로 변환
        const wav = new lamejs.WavHeader();
        wav.readHeader(new DataView(new Uint8Array(buffer).buffer));
        const samples = new Int16Array(buffer, wav.dataOffset, wav.dataLen / 2);
        const mp3Encoder = new lamejs.Mp3Encoder(wav.channels, wav.sampleRate, 128);
        const mp3Data = [];

        const sampleBlockSize = 1152; // LAME의 기본 샘플 블록 크기
        for (let i = 0; i < samples.length; i += sampleBlockSize) {
            const sampleChunk = samples.subarray(i, i + sampleBlockSize);
            const mp3buf = mp3Encoder.encodeBuffer(sampleChunk);
            if (mp3buf.length > 0) {
                mp3Data.push(new Int8Array(mp3buf));
            }
        }

        // 마지막 프레임 처리
        const finalMp3Buf = mp3Encoder.flush();
        if (finalMp3Buf.length > 0) {
            mp3Data.push(new Int8Array(finalMp3Buf));
        }

        // MP3 파일 생성
        const mp3Buffer = Buffer.concat(mp3Data);
        fs.writeFile(mp3FilePath, mp3Buffer, (err) => {
            if (err) {
                console.error(err);
                return res.status(500).send('Error writing MP3 file');
            }
            fs.unlinkSync(filePath); // 원본 파일 삭제
            res.send('Audio converted to MP3');
        });
    });
});

const port = 3000;
app.listen(port, () => {
    console.log(`Server running on port ${port}`);
});
