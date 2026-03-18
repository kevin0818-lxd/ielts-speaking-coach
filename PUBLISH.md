# 发布指南

## 发布前检查

- [ ] 更新 `clawhub.json` 中的 `homepage` 和 `support_url` 为你的 GitHub 仓库地址
- [ ] 在 `screenshots/` 目录添加 3–5 张截图（1920x1080 或 1280x720 PNG），可截取 Cursor 中 skill 评估输出界面
- [ ] 可选：录制 30–90 秒演示视频，上传到 YouTube/Vimeo，将 URL 加入 clawhub.json

## 步骤 1：发布到 GitHub

```bash
cd /Users/kevin/SpeakingCoachV1/ielts-speaking-coach-skill
git add -A && git commit -m "v1.1.0: audio pronunciation scoring, mock exam, ZPD learning paths"
git push origin main
```

## 步骤 2：发布到 ClawHub

```bash
clawhub publish ielts-speaking-coach-skill \
  --slug ielts-speaking-coach \
  --name "IELTS Speaking Coach" \
  --version 1.1.0 \
  --changelog "Add audio pronunciation scoring via ffmpeg+ASR, mock exam simulation, ZPD learning paths, adaptive difficulty, pronunciation guide, expanded cue cards."
```

## 权限说明（供 ClawHub 审核）

| 权限 | 用途 |
|------|------|
| network | 调用内置 LLM 能力、可选后端 API 通信 |
| shell | ffmpeg 音频格式转换 + ASR 语音识别，实现真实发音评分 |
