

// // 자동 스크롤 기능
// const messageContainer = document.getElementById('message-container');
// const scrollButton = document.getElementById('scroll-button');
// scrollButton.addEventListener('click', () => {
//   messageContainer.scrollTop = messageContainer.scrollHeight;
// });
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