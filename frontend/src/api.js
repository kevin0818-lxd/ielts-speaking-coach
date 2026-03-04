const BASE = '/api'

export async function uploadAudio(blob, part = 1, userId = 'default') {
  const ext = blob.type.includes('mp4') ? 'mp4' : blob.type.includes('ogg') ? 'ogg' : 'webm'
  const formData = new FormData()
  formData.append('file', blob, `recording.${ext}`)
  formData.append('part', String(part))
  formData.append('user_id', userId)

  const res = await fetch(`${BASE}/analyze_audio`, {
    method: 'POST',
    body: formData,
  })
  if (!res.ok) throw new Error(`Upload failed: ${res.status}`)
  return res.json()
}

export async function fetchTaskStatus(taskId) {
  const res = await fetch(`${BASE}/task_status/${taskId}`)
  if (!res.ok) throw new Error(`Poll failed: ${res.status}`)
  return res.json()
}

export function startPolling(taskId, { onUpdate, onComplete, onError, intervalMs = 2000, maxAttempts = 600 }) {
  let attempts = 0
  let stopped = false

  const poll = async () => {
    if (stopped) return
    attempts++
    try {
      const data = await fetchTaskStatus(taskId)
      if (stopped) return

      if (data.status === 'completed' && data.result) {
        onComplete(data.result)
        return
      }
      if (data.status === 'failed') {
        onError(new Error(data.error || 'Analysis failed on server.'))
        return
      }
      if (onUpdate) onUpdate(data.status, attempts)
      if (attempts >= maxAttempts) {
        onError(new Error('Analysis timed out. Please try again.'))
        return
      }
      setTimeout(poll, intervalMs)
    } catch (err) {
      if (!stopped) onError(err)
    }
  }

  poll()
  return () => { stopped = true }
}

export async function sendFeedback(original, recommended, action, userId = 'default') {
  const res = await fetch(`${BASE}/feedback_recommendation`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ original, recommended, action, user_id: userId }),
  })
  if (!res.ok) throw new Error(`Feedback failed: ${res.status}`)
  return res.json()
}

export async function playTTS(text) {
  const res = await fetch(`${BASE}/tts`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  })
  if (!res.ok) throw new Error(`TTS failed: ${res.status}`)
  const blob = await res.blob()
  const url = URL.createObjectURL(blob)
  const audio = new Audio(url)
  audio.onended = () => URL.revokeObjectURL(url)
  await audio.play()
  return audio
}
