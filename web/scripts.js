

// 자동 스크롤 기능
const messageContainer = document.getElementById('message-container');
const scrollButton = document.getElementById('scroll-button');
scrollButton.addEventListener('click', () => {
  messageContainer.scrollTop = messageContainer.scrollHeight;
});

// 녹음 및 메시지 전송
const recordButton = document.getElementById('record-button');
recordButton.addEventListener('click', () => {
  // MediaRecorder API 또는 유사한 기능을 사용하여 오디오 녹음
  // 오디오를 채팅 애플리케이션으로 전송
});
recordButton.addEventListener('click', () => {
  navigator.mediaDevices.getUserMedia({ audio: true })
    .then(stream => {
      const mediaRecorder = new MediaRecorder(stream);
      // 녹음 시작
      // 녹음이 멈추면, 오디오 데이터를 서버로 전송하거나 필요에 따라 처리
    })
    .catch(error => {
      console.error('허가 거부 또는 오류:', error);
    });
});


