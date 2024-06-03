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
            <p>{{ currentSentence }}</p>
          </div>
          <div class="display-result">
            <div class="score-result">
              <div class="circle">
                <div class="circle-inner">
                  <span class="score">{{ score }}%</span>
                </div>
              </div>
            </div>
          </div>
          <audio ref="audioPlayer" controls style="display: none;"></audio>
          <div id="waveform">

          </div>
          <div class="icon-container">
            <div class="volume" @click="playRecordedAudio">
              <IconVolume />
            </div>
            <div class="icon-mic" :class="{ 'recording': isRecording }" @click="toggleRecording">
              <IconMic />
            </div>
            <div class="ear" @click="playAudio">
              <IconEar />
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

<script>
import IconMic from '@/components/icons/IconMic.vue';
import IconEar from '@/components/icons/IconEar.vue';
import IconVolume from '@/components/icons/IconVolume.vue';
import axios from 'axios';
import WaveSurfer from 'wavesurfer.js';

export default {
  name: 'HomeView',
  components: {
    IconMic,
    IconEar,
    IconVolume,
  },
  data() {
    return {
      apiData:[],
      text:[],
      audio: [],
      currentIndex: 0,
      progress: 0,
      isRecording: false,
      recordedAudio: null,
      mediaRecorder: null,
      recordedChunks: [],
      audioSaved: false,
      activeButton: null,
      score: 70,
      getalldata: false,
      length_data: 0,
      wavesurfer: null,
      animationFrameId: null,
    };
  },
  computed: {
    currentSentence() {
      return this.text[this.currentIndex] || '';
    }
  },
  created() {
    this.fetchData();
  },
  mounted() {
    this.initWaveSurfer();
  },
  methods: {
    initWaveSurfer() {
      this.wavesurfer = WaveSurfer.create({
        container: '#waveform',
        waveColor: 'red',
        progressColor: '#4CAF50',
        cursorWidth: 0,
        height:100,
        responsive: true,
        backend: 'WebAudio',
      });

      this.wavesurfer.on('ready', () => {
        console.log('WaveSurfer is ready');
      });

      this.wavesurfer.on('error', (e) => {
        console.error(e);
      });
    },
    setActive(button) {
      this.activeButton = button;
    },
    playRecordedAudio() {
      if (this.recordedChunks.length > 0) {
        let recordedBlob = new Blob(this.recordedChunks, { type: 'audio/ogg' });
        const audioUrl = URL.createObjectURL(recordedBlob);
        this.$refs.audioPlayer.src = audioUrl;
        this.$refs.audioPlayer.play();
        this.wavesurfer.load(audioUrl);

        this.wavesurfer.on('ready', () => {
          this.wavesurfer.play();
          this.startAnimation();
        });

        this.wavesurfer.on('finish', () => {
          this.stopAnimation();
        });
      } else {
        console.error('Không có file âm thanh ghi được.');
      }
    },
    toggleRecording() {
      this.isRecording = !this.isRecording;
      if (this.isRecording) {
        this.startRecording();
      } else {
        this.stopRecording();
      }
    },
    startRecording() {
        navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
          this.mediaRecorder = new MediaRecorder(stream);
          this.recordedChunks = [];
          this.mediaRecorder.ondataavailable = event => {
            this.recordedChunks.push(event.data);
          };
          this.mediaRecorder.start();
          document.querySelector('.icon-mic').style.border = "3px solid red";
        })
        .catch(error => {
          console.error('Lỗi truy cập microphone:', error);
        });
    },
    stopRecording() {
      if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
        this.mediaRecorder.stop();
        document.querySelector('.icon-mic').style.border = "1px solid black";
      }

      this.mediaRecorder.onstop = async () => {this.sendAudio()};
    },
    async sendAudio() {
      const convertBlobToBase64 = async (blob) => {
        return await blobToBase64(blob);
      }

      const blobToBase64 = blob => new Promise((resolve, reject) => {
          const reader = new FileReader();
          reader.readAsDataURL(blob);
          reader.onload = () => resolve(reader.result);
          reader.onerror = error => reject(error);
      });

      console.log(this.recordedChunks.length)

      if (this.recordedChunks.length > 0) {
        let audioBlob = new Blob(this.recordedChunks, { type: 'audio/ogg' });

        let audioUrl = URL.createObjectURL(audioBlob);
        let audioRecorded = new Audio(audioUrl);

        let audioBase64 = await convertBlobToBase64(audioBlob);

        let minimumAllowedLength = 6;
        if (audioBase64.length < minimumAllowedLength) {
            setTimeout(UIRecordingError, 50); // Make sure this function finished after get called again
            return;
        }

        try {
          const response = await axios.post('/api/GetAccuracyFromRecordedAudio', JSON.stringify({ "title": "how old are you", "base64Audio": audioBase64}), {
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
    },
    async fetchData() {
      try {
        const response = await axios.get('/api/getSample');
        this.text = response.data.text;
        this.ipa = response.data.ipa;
        this.audio = response.data.audio;
        this.length_data = this.text.length;
        console.log(this.text, this.audio);
      } catch (error) {
        console.error('Lỗi khi lấy dữ liệu:', error);
      }
    },
    nextSentence() {
      if (this.currentIndex < this.text.length - 1) {
        this.currentIndex++;
        this.progress = (this.currentIndex / this.text.length) * 100;
      }
    },
    previousSentence() {
      if (this.currentIndex > 0) {
        this.currentIndex--;
        this.progress = (this.currentIndex / this.text.length) * 100;
      }
    },
    updateProgress() {
      const currentTime = this.wavesurfer.getCurrentTime();
      const duration = this.wavesurfer.getDuration();
    },
    startAnimation() {
      const animate = () => {
        this.updateProgress();
        this.animationFrameId = requestAnimationFrame(animate);
      };
      this.animationFrameId = requestAnimationFrame(animate);
    },
    stopAnimation() {
      if (this.animationFrameId) {
        cancelAnimationFrame(this.animationFrameId);
        this.animationFrameId = null;
      }
    },
  },
  beforeDestroy() {
    if (this.wavesurfer) {
      this.wavesurfer.destroy();
    }
    this.stopAnimation();
  },
  playAudio() {
      if (this.audio.length > 0 && this.currentIndex >= 0 && this.currentIndex < this.audio.length) {
        const audioUrl = `data:audio/wav;base64,${this.audio[this.currentIndex]}`;
        const audio = new Audio(audioUrl);
        audio.play();
        console.log('phát âm thanh');
      } else {
        console.error('Không có file âm thanh hoặc index không hợp lệ.');
      }
    },
};
</script>
<style scoped>
/* CSS cho giao diện */
@import url('./../assets/home.css');
</style>
