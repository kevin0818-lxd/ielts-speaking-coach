<template>
  <div class="grammar-card" v-if="item">
    <span :class="['type-badge', item.type === 'Correction' ? 'correction' : 'enhance']">
      {{ item.type || 'Correction' }}
    </span>
    <div class="grammar-original" v-if="item.original">"{{ item.original }}"</div>
    <div class="grammar-suggestion" v-if="item.suggestion">
      <strong>Suggestion:</strong> {{ item.suggestion }}
    </div>
    <div class="grammar-explanation" v-if="item.explanation">{{ item.explanation }}</div>
  </div>
</template>

<script setup>
defineProps({ item: Object })
</script>

<style scoped>
.grammar-card {
  background: #fff;
  border: 1px solid #eee;
  border-radius: 10px;
  padding: 14px;
  margin-bottom: 10px;
}
.type-badge {
  display: inline-block;
  font-size: 11px;
  font-weight: 700;
  padding: 2px 10px;
  border-radius: 10px;
  margin-bottom: 8px;
  text-transform: uppercase;
}
.correction { background: #fdecea; color: #ee0a24; }
.enhance { background: #fff5eb; color: #ff976a; }
.grammar-original {
  font-weight: 600;
  color: #333;
  font-size: 14px;
  margin-bottom: 4px;
}
.grammar-suggestion {
  font-size: 13px;
  color: #07c160;
  margin-bottom: 4px;
}
.grammar-explanation {
  font-size: 12px;
  color: #888;
  line-height: 1.5;
}
</style>
