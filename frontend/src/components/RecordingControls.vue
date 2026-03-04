<template>
  <div class="controls">
    <div v-if="isRecording" class="recording-active">
      <div class="pulse-ring"></div>
      <div class="timer">{{ formatTime(elapsedSeconds) }}</div>
      <div class="rec-hint">Recording...</div>
      <button class="btn btn-stop" @click="$emit('stop')">Stop</button>
    </div>

    <div v-else-if="hasAudio" class="post-record">
      <div class="timer dim">{{ formatTime(elapsedSeconds) }}</div>
      <div class="action-row">
        <button class="btn btn-outline" @click="togglePlay">
          {{ playing ? 'Pause' : 'Play Preview' }}
        </button>
        <button class="btn btn-outline" @click="$emit('retry')">Retry</button>
        <button class="btn btn-primary" :disabled="analyzing" @click="$emit('analyze')">
          {{ analyzing ? 'Analyzing...' : 'Analyze Response' }}
        </button>
      </div>
    </div>

    <div v-else class="idle">
      <button class="btn btn-record" @click="$emit('start')">
        <span class="mic-icon">&#127908;</span>
        Start Recording
      </button>
      <div class="idle-hint">Tap to record your IELTS Speaking response (max 2 min)</div>
    </div>

    <audio ref="audioEl" :src="audioUrl" @ended="playing = false"></audio>
  </div>
</template>

<script setup>
import { ref } from 'vue'

defineProps({
  isRecording: Boolean,
  hasAudio: Boolean,
  audioUrl: String,
  elapsedSeconds: { type: Number, default: 0 },
  analyzing: Boolean,
})
defineEmits(['start', 'stop', 'retry', 'analyze'])

const audioEl = ref(null)
const playing = ref(false)

function togglePlay() {
  if (!audioEl.value) return
  if (playing.value) {
    audioEl.value.pause()
    playing.value = false
  } else {
    audioEl.value.play()
    playing.value = true
  }
}

function formatTime(s) {
  const m = Math.floor(s / 60)
  const sec = s % 60
  return `${m}:${String(sec).padStart(2, '0')}`
}
</script>

<style scoped>
.controls {
  text-align: center;
  padding: 24px 0;
}
.timer {
  font-size: 36px;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
  color: #333;
}
.timer.dim { color: #999; }
.rec-hint {
  color: #ee0a24;
  font-size: 14px;
  margin: 4px 0 16px;
  font-weight: 500;
}
.idle-hint {
  color: #999;
  font-size: 13px;
  margin-top: 12px;
}
.action-row {
  display: flex;
  gap: 10px;
  justify-content: center;
  flex-wrap: wrap;
  margin-top: 16px;
}
.btn {
  padding: 10px 24px;
  border-radius: 24px;
  font-size: 14px;
  font-weight: 600;
  border: none;
  cursor: pointer;
  transition: all 0.2s;
}
.btn:disabled { opacity: 0.5; cursor: not-allowed; }
.btn-record {
  background: #ee0a24;
  color: #fff;
  padding: 16px 32px;
  font-size: 16px;
  border-radius: 32px;
}
.btn-record:hover { background: #d00920; }
.mic-icon { margin-right: 6px; }
.btn-stop {
  background: #ee0a24;
  color: #fff;
}
.btn-outline {
  background: #fff;
  color: #333;
  border: 2px solid #ddd;
}
.btn-outline:hover { border-color: #07c160; color: #07c160; }
.btn-primary {
  background: #07c160;
  color: #fff;
}
.btn-primary:hover { background: #06ad56; }
.pulse-ring {
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: #ee0a24;
  margin: 0 auto 12px;
  animation: pulse 1.2s infinite;
}
@keyframes pulse {
  0% { box-shadow: 0 0 0 0 rgba(238, 10, 36, 0.5); }
  70% { box-shadow: 0 0 0 14px rgba(238, 10, 36, 0); }
  100% { box-shadow: 0 0 0 0 rgba(238, 10, 36, 0); }
}
.recording-active { padding: 8px 0; }
.post-record { padding: 8px 0; }
</style>
