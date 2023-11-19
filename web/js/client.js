// 클라이언트 측 JavaScript
let mediaRecorder;
let audioChunks = [];

// 녹음 시작
function startRecording(stream) {
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.start();
    mediaRecorder.ondataavailable = event => {
        audioChunks.push(event.data);
    };
}

// 녹음 중지 및 서버로 전송
function stopRecording() {
    mediaRecorder.stop();
    mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
        const formData = new FormData();
        formData.append('audioFile', audioBlob);

        try {
            const response = await fetch('http://localhost:3000/upload', {
                method: 'POST',
                body: formData
            });
            // 서버로부터의 응답 처리
            console.log('Audio uploaded');
        } catch (error) {
            console.error('Upload error:', error);
        }
    };
}
