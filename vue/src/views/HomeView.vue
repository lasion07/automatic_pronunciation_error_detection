<template>
  <div class="home-container">
    <div class="main-element">
      <div class="main-container">
        <div class="main-content">
          <div class="header">
            <div class="process">
              <div class="icon-x">
                <!-- <IconX/> -->
              </div>
              <div class="progress-tool">
                <div class="progress-container">
                  <div class="progress-bar" :style="{ width: progress + '%', height: '100%', backgroundColor: 'green', position: 'absolute' }"></div>
                </div>
              </div>
            </div>
          </div>
          <!-- <p>READ THE SENTENCES ALOUD</p> -->
          <div class="content">
            <div id="sentence">
              <span v-for="(wordObj, index) in coloredWords" :key="index" :class="wordObj.color">
                {{ wordObj.word }}
                <span v-if="index !== coloredWords.length - 1">&nbsp;</span>
              </span>
            </div>
            <!--transcription-->
            <div id="sentence">
              <span v-for="(word, index) in currentSentence.split(' ')" :key="index">
                {{ word }}
                <span v-if="index !== currentSentence.split(' ').length - 1">&nbsp;</span>
              </span>
            </div>
          </div>
          <audio ref="audioPlayer" controls style="display: none;"></audio>
          <div id="waveform"></div>
          <div class="icon-container">
            <div id="play_user_record_btn" class="button volume" @click="playRecordedAudio">
              <IconVolume />
            </div>
            <div id="record_btn" class="button icon-mic" :class="{ 'recording': isRecording }" @click="toggleRecording">
              <IconMic />
            </div>
            <div id="play_sample_record_btn" class="button ear" @click="playAudio">
              <IconEar />
            </div>
          </div>
        </div>
      </div>
      <div class="display-result">
        <div class="score-result">
          <div class="circle" :style="{ '--percentage': percentage + '%' }">
            <div class="circle-inner">
              <span class="score">{{ score }}%</span>
            </div>
          </div>
        </div>
      </div>
    </div>
    <div class="footer">
      <div class="divider"></div>
      <div class="button-container">
        <button @click="previousSentence">PREVIOUS</button>
        <button @click="nextSentence">NEXT</button>
      </div>
    </div>
  </div>
</template>
<script setup>
import { ref, computed, onMounted, onBeforeUnmount } from 'vue';
import axios from 'axios';
import WaveSurfer from 'wavesurfer.js';
import IconMic from '@/components/icons/IconMic.vue';
import IconEar from '@/components/icons/IconEar.vue';
import IconVolume from '@/components/icons/IconVolume.vue';

const apiData = ref([]);
const text = ref([]);
const audio = ref([]);
const currentIndex = ref(0);
const progress = ref(0);
const isRecording = ref(false);
const recordedAudio = ref(null);
const mediaRecorder = ref(null);
const recordedChunks = ref([]);
const audioSaved = ref(false);
const activeButton = ref(null);
// const score = ref(90);
const getalldata = ref(false);
const length_data = ref(0);
const wavesurfer = ref(null);
const animationFrameId = ref(null);
// demo  =================================
const score = ref(0);
const percentage = ref(0);
const sentence = "hold your nose to keep the smell from disabling your motor functions";
const coloredWords = ref([]);

generateColoredWords();

function generateColoredWords() {
  const words = sentence.split(' ');
  for (let i = 0; i < words.length; i++) {
    const score = Math.random();
    let color;
    if (score >= 0.8) {
      color = 'green';
    } else if (score < 0.5) {
      color = 'red';
    } else {
      color = 'yellow';
    }
    coloredWords.value.push({ word: words[i], color: color });
  }
}

const animateScore = (start, end, duration) => {
  const stepTime = Math.abs(Math.floor(duration / (end - start)));
  let current = start;
  const increment = end > start ? 1 : -1;
  const timer = setInterval(() => {
    current += increment;
    percentage.value = current;
    if (current === end) {
      clearInterval(timer);
    }
  }, stepTime);
};

// // Thiết lập giá trị của score 
onMounted(() => {
  score.value = 50; // Giá trị score 
  animateScore(0, score.value, 800);
});
// ============================================
const currentSentence = computed(() => {
  return text.value[currentIndex.value] || '';
});

const fetchData = async () => {
  try {
    const response = await axios.get('/api/getall');
    text.value = response.data.text;
    audio.value = response.data.audio;
    length_data.value = text.value.length;
    console.log(text.value, audio.value);
  } catch (error) {
    console.error('Lỗi khi lấy dữ liệu:', error);
  }
};

const initWaveSurfer = () => {
  wavesurfer.value = WaveSurfer.create({
    container: '#waveform',
    waveColor: 'red',
    progressColor: '#4CAF50',
    cursorWidth: 0,
    height: 100,
    responsive: true,
    backend: 'WebAudio',
  });

  wavesurfer.value.on('ready', () => {
    console.log('WaveSurfer is ready');
  });
  wavesurfer.value.on('error', (e) => {
    console.error(e);
  });
};

const setActive = (button) => {
  activeButton.value = button;
};

const playRecordedAudio = () => {
  if (recordedChunks.value.length > 0) {
    let recordedBlob = new Blob(recordedChunks.value, { type: 'audio/ogg' });
    const audioUrl = URL.createObjectURL(recordedBlob);
    const audioPlayer = new Audio(audioUrl);
    audioPlayer.play();
    wavesurfer.value.load(audioUrl);
    wavesurfer.value.on('ready', () => {
      wavesurfer.value.play();
      startAnimation();
    });

    wavesurfer.value.on('finish', () => {
      stopAnimation();
    });
  } else {
    console.error('Không có file âm thanh ghi được.');
  }
};

const toggleRecording = () => {
  isRecording.value = !isRecording.value;
  if (isRecording.value) {
    startRecording();
  } else {
    stopRecording();
  }
};

const startRecording = () => {
  navigator.mediaDevices.getUserMedia({ audio: true })
    .then(stream => {
      mediaRecorder.value = new MediaRecorder(stream);
      recordedChunks.value = [];
      mediaRecorder.value.ondataavailable = event => {
        recordedChunks.value.push(event.data);
      };
      mediaRecorder.value.start();
    })
    .catch(error => {
      console.error('Lỗi truy cập microphone:', error);
    });
};

const stopRecording = () => {
  if (mediaRecorder.value && mediaRecorder.value.state !== 'inactive') {
    mediaRecorder.value.stop();
  }

  mediaRecorder.value.onstop = async () => { sendAudio() };
};

const sendAudio = async () => {
  const convertBlobToBase64 = async (blob) => {
    return await blobToBase64(blob);
  };

  const blobToBase64 = blob => new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(blob);
    reader.onload = () => resolve(reader.result);
    reader.onerror = error => reject(error);
  });

  if (recordedChunks.value.length > 0) {
    let audioBlob = new Blob(recordedChunks.value, { type: 'audio/ogg' });
    let audioBase64 = await convertBlobToBase64(audioBlob);

    let minimumAllowedLength = 6;
    if (audioBase64.length < minimumAllowedLength) {
      setTimeout(() => { console.error('File audio quá ngắn.'); }, 50);
      return;
    }

    try {
      const response = await axios.post('/api/GetAccuracyFromRecordedAudio', JSON.stringify({ "title": "how old are you", "base64Audio": audioBase64 }), {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      console.log('File âm thanh đã được gửi thành công.', response);
    } catch (error) {
      console.error('Lỗi khi gửi file âm thanh:', error);
    }
  } else {
    console.error('Không gửi được file âm thanh.');
  }
};

const nextSentence = () => {
  if (currentIndex.value < text.value.length - 1) {
    currentIndex.value++;
    progress.value = (currentIndex.value / (text.value.length-1)) * 100;
    colors.value = 'red';
  }
};

const previousSentence = () => {
  if (currentIndex.value > 0) {
    currentIndex.value--;
    progress.value = (currentIndex.value / (text.value.length-1)) * 100;
    colors.value = 'green';
  }
};

const updateProgress = () => {
  const currentTime = wavesurfer.value.getCurrentTime();
  const duration = wavesurfer.value.getDuration();
};

const startAnimation = () => {
  const animate = () => {
    updateProgress();
    animationFrameId.value = requestAnimationFrame(animate);
  };
  animationFrameId.value = requestAnimationFrame(animate);
};

const stopAnimation = () => {
  if (animationFrameId.value) {
    cancelAnimationFrame(animationFrameId.value);
    animationFrameId.value = null;
  }
};

onMounted(() => {
  fetchData();
  initWaveSurfer();
});

onBeforeUnmount(() => {
  if (wavesurfer.value) {
    wavesurfer.value.destroy();
  }
  stopAnimation();
});
</script>

<style scoped>
@import url('./../assets/home.css');
</style>
