

// 자동 스크롤 기능
const messageContainer = document.getElementById('message-container');
const scrollButton = document.getElementById('scroll-button');
scrollButton.addEventListener('click', () => {
  messageContainer.scrollTop = messageContainer.scrollHeight;
});

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
                mediaRecorder.start();
                audioChunks = [];

                mediaRecorder.ondataavailable = function(event) {
                    audioChunks.push(event.data);
                };

                mediaRecorder.onstop = function() {
                    // 여기서 audioChunks를 MP3로 변환하고 저장하는 로직을 추가합니다.
                };
            });
    }
});


// 녹음 시작
function startRecording(stream) {
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.start();
    mediaRecorder.ondataavailable = event => {
        audioChunks.push(event.data);
    };
}

