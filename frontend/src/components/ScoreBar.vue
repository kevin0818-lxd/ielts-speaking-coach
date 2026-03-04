<template>
  <div class="score-bar">
    <div class="score-bar-header">
      <span class="score-bar-label">{{ label }}</span>
      <span class="score-bar-value">{{ value.toFixed(1) }}</span>
    </div>
    <div class="score-bar-track">
      <div class="score-bar-fill" :style="{ width: pct + '%', background: color }"></div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  label: String,
  value: { type: Number, default: 0 },
  color: { type: String, default: '#07c160' },
})

const pct = computed(() => Math.min(100, Math.max(0, (props.value / 9) * 100)))
</script>

<style scoped>
.score-bar { margin: 8px 0; }
.score-bar-header {
  display: flex;
  justify-content: space-between;
  font-size: 13px;
  margin-bottom: 4px;
}
.score-bar-label { color: #666; font-weight: 500; }
.score-bar-value { font-weight: 700; color: #333; }
.score-bar-track {
  height: 8px;
  border-radius: 4px;
  background: #eee;
  overflow: hidden;
}
.score-bar-fill {
  height: 100%;
  border-radius: 4px;
  transition: width 0.6s ease;
}
</style>
