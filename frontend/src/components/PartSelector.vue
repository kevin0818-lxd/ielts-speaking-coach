<template>
  <div class="part-selector">
    <span class="part-label">Speaking Part</span>
    <div class="part-options">
      <button
        v-for="p in parts"
        :key="p.value"
        :class="['part-btn', { active: modelValue === p.value }]"
        @click="$emit('update:modelValue', p.value)"
      >
        {{ p.label }}
      </button>
    </div>
  </div>
</template>

<script setup>
defineProps({ modelValue: { type: Number, default: 1 } })
defineEmits(['update:modelValue'])

const parts = [
  { value: 1, label: 'Part 1' },
  { value: 2, label: 'Part 2' },
  { value: 3, label: 'Part 3' },
]
</script>

<style scoped>
.part-selector {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 0;
}
.part-label {
  font-weight: 600;
  font-size: 14px;
  color: #333;
  white-space: nowrap;
}
.part-options {
  display: flex;
  gap: 8px;
}
.part-btn {
  padding: 8px 20px;
  border: 2px solid #ddd;
  border-radius: 20px;
  background: #fff;
  font-size: 14px;
  font-weight: 500;
  color: #666;
  cursor: pointer;
  transition: all 0.2s;
}
.part-btn:hover {
  border-color: #07c160;
  color: #07c160;
}
.part-btn.active {
  background: #07c160;
  border-color: #07c160;
  color: #fff;
}
</style>
