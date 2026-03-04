import { ref, onUnmounted } from 'vue'

const MAX_DURATION_MS = 120_000

function pickMimeType() {
  const candidates = [
    'audio/webm;codecs=opus',
    'audio/webm',
    'audio/ogg;codecs=opus',
    'audio/mp4',
  ]
  for (const mime of candidates) {
    if (typeof MediaRecorder !== 'undefined' && MediaRecorder.isTypeSupported(mime)) {
      return mime
    }
  }
  return ''
}

export function useRecorder() {
  const isRecording = ref(false)
  const hasAudio = ref(false)
  const audioBlob = ref(null)
  const audioUrl = ref('')
  const elapsedSeconds = ref(0)

  let mediaRecorder = null
  let chunks = []
  let stream = null
  let timerInterval = null
  let autoStopTimer = null

  function clearTimers() {
    if (timerInterval) { clearInterval(timerInterval); timerInterval = null }
    if (autoStopTimer) { clearTimeout(autoStopTimer); autoStopTimer = null }
  }

  async function startRecording() {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    } catch {
      throw new Error('Microphone access denied. Please allow microphone permissions.')
    }

    chunks = []
    const mimeType = pickMimeType()
    const options = mimeType ? { mimeType } : {}
    mediaRecorder = new MediaRecorder(stream, options)

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunks.push(e.data)
    }

    mediaRecorder.onstop = () => {
      clearTimers()
      const blob = new Blob(chunks, { type: mimeType || 'audio/webm' })
      audioBlob.value = blob
      audioUrl.value = URL.createObjectURL(blob)
      hasAudio.value = true
      isRecording.value = false
      if (stream) {
        stream.getTracks().forEach((t) => t.stop())
        stream = null
      }
    }

    mediaRecorder.start(250)
    isRecording.value = true
    hasAudio.value = false
    elapsedSeconds.value = 0

    timerInterval = setInterval(() => { elapsedSeconds.value++ }, 1000)
    autoStopTimer = setTimeout(() => stopRecording(), MAX_DURATION_MS)
  }

  function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
      mediaRecorder.stop()
    }
    clearTimers()
  }

  function resetRecording() {
    stopRecording()
    if (audioUrl.value) URL.revokeObjectURL(audioUrl.value)
    audioBlob.value = null
    audioUrl.value = ''
    hasAudio.value = false
    elapsedSeconds.value = 0
    isRecording.value = false
  }

  onUnmounted(() => {
    resetRecording()
  })

  return {
    isRecording,
    hasAudio,
    audioBlob,
    audioUrl,
    elapsedSeconds,
    startRecording,
    stopRecording,
    resetRecording,
  }
}
