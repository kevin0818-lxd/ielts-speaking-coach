<template>
  <div class="vocab-card" v-if="item">
    <div class="vocab-original">"{{ item.original }}"</div>
    <div class="vocab-arrow">&#8594;</div>
    <div class="vocab-alternatives">
      <span v-for="(alt, i) in (item.alternatives || [])" :key="i" class="alt-chip">{{ alt }}</span>
    </div>
    <div class="vocab-reason" v-if="item.reason">{{ item.reason }}</div>
    <div class="vocab-actions">
      <button class="action-btn accept" @click="$emit('feedback', item.original, (item.alternatives||[])[0], 'accept')">Accept</button>
      <button class="action-btn ignore" @click="$emit('feedback', item.original, (item.alternatives||[])[0], 'ignore')">Ignore</button>
    </div>
  </div>
</template>

<script setup>
defineProps({ item: Object })
defineEmits(['feedback'])
</script>

<style scoped>
.vocab-card {
  background: #fff;
  border: 1px solid #eee;
  border-radius: 10px;
  padding: 14px;
  margin-bottom: 10px;
}
.vocab-original {
  font-weight: 700;
  color: #ee0a24;
  font-size: 14px;
}
.vocab-arrow {
  color: #ccc;
  font-size: 18px;
  margin: 4px 0;
}
.vocab-alternatives {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: 8px;
}
.alt-chip {
  background: #e8f8ee;
  color: #07c160;
  padding: 3px 10px;
  border-radius: 12px;
  font-size: 13px;
  font-weight: 600;
}
.vocab-reason {
  font-size: 12px;
  color: #888;
  margin-bottom: 8px;
  line-height: 1.5;
}
.vocab-actions {
  display: flex;
  gap: 8px;
}
.action-btn {
  font-size: 12px;
  padding: 4px 14px;
  border-radius: 12px;
  border: none;
  cursor: pointer;
  font-weight: 600;
}
.accept { background: #07c160; color: #fff; }
.ignore { background: #eee; color: #666; }
</style>
