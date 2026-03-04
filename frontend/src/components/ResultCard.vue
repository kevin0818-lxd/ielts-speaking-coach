<template>
  <div class="result" v-if="data">
    <!-- Overall Score -->
    <div class="score-hero">
      <div class="score-number">{{ data.score?.toFixed(1) || '—' }}</div>
      <div class="score-label">Overall Band</div>
    </div>

    <!-- Breakdown -->
    <div class="card" v-if="data.breakdown">
      <div class="card-title">Score Breakdown</div>
      <ScoreBar label="Fluency & Coherence" :value="data.breakdown.fluency || 0" color="#1989fa" />
      <ScoreBar label="Lexical Resource" :value="data.breakdown.lexical || 0" color="#07c160" />
      <ScoreBar label="Grammar" :value="data.breakdown.grammar || 0" color="#ff976a" />
      <ScoreBar label="Pronunciation" :value="data.breakdown.pronunciation || 0" color="#ee0a24" />
    </div>

    <!-- Model Answer -->
    <ModelAnswer :text="data.model_answer || ''" />

    <!-- Rationale -->
    <div class="card" v-if="data.rationale && data.rationale.length">
      <div class="card-title">Feedback Details</div>
      <ul class="rationale-list">
        <li v-for="(r, i) in data.rationale" :key="i" v-html="r"></li>
      </ul>
    </div>

    <!-- Vocabulary -->
    <div class="card" v-if="data.recommendations && data.recommendations.length">
      <div class="card-title">Vocabulary Upgrade</div>
      <VocabCard
        v-for="(rec, i) in data.recommendations"
        :key="i"
        :item="rec"
        @feedback="onFeedback"
      />
    </div>

    <!-- Grammar -->
    <div class="card" v-if="data.grammar_recommendations && data.grammar_recommendations.length">
      <div class="card-title">Grammar Coach</div>
      <GrammarCard
        v-for="(g, i) in data.grammar_recommendations"
        :key="i"
        :item="g"
      />
    </div>

    <!-- Transcript -->
    <div class="card" v-if="data.transcription">
      <div class="card-title">Your Transcript</div>
      <div class="transcript-text">{{ data.transcription }}</div>
    </div>
  </div>
</template>

<script setup>
import ScoreBar from './ScoreBar.vue'
import ModelAnswer from './ModelAnswer.vue'
import VocabCard from './VocabCard.vue'
import GrammarCard from './GrammarCard.vue'
import { sendFeedback } from '../api.js'

defineProps({ data: { type: Object, default: null } })

async function onFeedback(original, recommended, action) {
  try {
    await sendFeedback(original, recommended, action)
  } catch {
    /* silently ignore feedback errors */
  }
}
</script>

<style scoped>
.score-hero {
  text-align: center;
  padding: 24px 0 16px;
}
.score-number {
  font-size: 56px;
  font-weight: 800;
  color: #07c160;
  line-height: 1;
}
.score-label {
  font-size: 14px;
  color: #888;
  margin-top: 4px;
}
.card {
  background: #fff;
  border-radius: 12px;
  padding: 16px;
  margin-top: 16px;
  box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.card-title {
  font-weight: 700;
  font-size: 15px;
  color: #333;
  margin-bottom: 12px;
}
.rationale-list {
  padding-left: 20px;
  margin: 0;
  font-size: 13px;
  line-height: 1.8;
  color: #555;
}
.transcript-text {
  font-size: 14px;
  line-height: 1.7;
  color: #555;
  white-space: pre-wrap;
}
</style>
