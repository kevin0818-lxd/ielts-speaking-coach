<template>
  <div class="model-answer" v-if="text">
    <div class="section-title">
      <span>Model Answer</span>
      <button class="tts-btn" @click="speak" :disabled="speaking">
        {{ speaking ? 'Playing...' : 'Listen' }}
      </button>
    </div>
    <div class="model-text">{{ text }}</div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { playTTS } from '../api.js'

const props = defineProps({ text: { type: String, default: '' } })

const speaking = ref(false)

async function speak() {
  if (!props.text || speaking.value) return
  speaking.value = true
  try {
    const audio = await playTTS(props.text)
    audio.onended = () => { speaking.value = false }
  } catch {
    speaking.value = false
  }
}
</script>

<style scoped>
.model-answer {
  background: #e8f8ee;
  border-radius: 12px;
  padding: 16px;
  margin-top: 16px;
}
.section-title {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-weight: 700;
  font-size: 15px;
  color: #07c160;
  margin-bottom: 10px;
}
.tts-btn {
  font-size: 12px;
  padding: 4px 14px;
  border-radius: 14px;
  border: 1.5px solid #07c160;
  background: #fff;
  color: #07c160;
  cursor: pointer;
  font-weight: 600;
}
.tts-btn:disabled { opacity: 0.5; cursor: not-allowed; }
.model-text {
  font-size: 14px;
  line-height: 1.7;
  color: #333;
  white-space: pre-wrap;
}
</style>
