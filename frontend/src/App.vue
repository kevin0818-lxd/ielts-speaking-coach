<template>
  <div class="app">
    <header class="app-header">
      <h1>IELTS Speaking Coach</h1>
      <p class="subtitle">AI-Powered Scoring &amp; Feedback</p>
    </header>

    <main class="app-body">
      <PartSelector v-model="selectedPart" />

      <RecordingControls
        :isRecording="recorder.isRecording.value"
        :hasAudio="recorder.hasAudio.value"
        :audioUrl="recorder.audioUrl.value"
        :elapsedSeconds="recorder.elapsedSeconds.value"
        :analyzing="analyzing"
        @start="onStart"
        @stop="recorder.stopRecording()"
        @retry="onRetry"
        @analyze="onAnalyze"
      />

      <!-- Loading overlay -->
      <div v-if="analyzing" class="loading-overlay">
        <div class="spinner"></div>
        <div class="loading-text">{{ statusText }}</div>
      </div>

      <!-- Error -->
      <div v-if="errorMsg" class="error-banner">
        {{ errorMsg }}
        <button class="dismiss-btn" @click="errorMsg = ''">Dismiss</button>
      </div>

      <!-- Result -->
      <ResultCard :data="result" />
    </main>

    <footer class="app-footer">
      Powered by AI &middot; Kevin's IELTS Speaking Coach
    </footer>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import PartSelector from './components/PartSelector.vue'
import RecordingControls from './components/RecordingControls.vue'
import ResultCard from './components/ResultCard.vue'
import { useRecorder } from './composables/useRecorder.js'
import { uploadAudio, startPolling } from './api.js'

const selectedPart = ref(1)
const analyzing = ref(false)
const statusText = ref('Uploading...')
const errorMsg = ref('')
const result = ref(null)

const recorder = useRecorder()
let stopPolling = null

async function onStart() {
  errorMsg.value = ''
  result.value = null
  try {
    await recorder.startRecording()
  } catch (err) {
    errorMsg.value = err.message
  }
}

function onRetry() {
  recorder.resetRecording()
  result.value = null
  errorMsg.value = ''
}

async function onAnalyze() {
  if (!recorder.audioBlob.value) return
  analyzing.value = true
  statusText.value = 'Uploading audio...'
  errorMsg.value = ''
  result.value = null

  try {
    const uploadRes = await uploadAudio(recorder.audioBlob.value, selectedPart.value)

    if (uploadRes.task_id) {
      statusText.value = 'Analyzing your response...'
      stopPolling = startPolling(uploadRes.task_id, {
        onUpdate(status, attempt) {
          statusText.value = `Analyzing... (${status}, attempt ${attempt})`
        },
        onComplete(data) {
          result.value = data
          analyzing.value = false
        },
        onError(err) {
          errorMsg.value = err.message || 'Analysis failed.'
          analyzing.value = false
        },
      })
    } else if (uploadRes.error) {
      errorMsg.value = uploadRes.error
      analyzing.value = false
    } else {
      result.value = uploadRes
      analyzing.value = false
    }
  } catch (err) {
    errorMsg.value = err.message || 'Upload failed. Please try again.'
    analyzing.value = false
  }
}
</script>

<style>
/* Global reset applied here, scoped styles in next todo */
*, *::before, *::after {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
  background: #f7f8fa;
  color: #333;
  -webkit-font-smoothing: antialiased;
}
</style>

<style scoped>
.app {
  max-width: 480px;
  margin: 0 auto;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}
.app-header {
  text-align: center;
  padding: 28px 16px 12px;
  background: linear-gradient(135deg, #07c160 0%, #06ad56 100%);
  color: #fff;
  border-radius: 0 0 20px 20px;
}
.app-header h1 {
  font-size: 22px;
  font-weight: 800;
  letter-spacing: -0.3px;
}
.subtitle {
  font-size: 13px;
  opacity: 0.85;
  margin-top: 4px;
}
.app-body {
  flex: 1;
  padding: 0 16px 24px;
}
.loading-overlay {
  text-align: center;
  padding: 32px 0;
}
.spinner {
  width: 36px;
  height: 36px;
  border: 4px solid #eee;
  border-top-color: #07c160;
  border-radius: 50%;
  margin: 0 auto 12px;
  animation: spin 0.8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
.loading-text {
  font-size: 14px;
  color: #888;
}
.error-banner {
  background: #fdecea;
  color: #ee0a24;
  padding: 12px 16px;
  border-radius: 10px;
  font-size: 13px;
  margin-top: 12px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.dismiss-btn {
  background: none;
  border: none;
  color: #ee0a24;
  font-weight: 600;
  cursor: pointer;
  font-size: 12px;
}
.app-footer {
  text-align: center;
  padding: 16px;
  font-size: 12px;
  color: #bbb;
}
</style>
