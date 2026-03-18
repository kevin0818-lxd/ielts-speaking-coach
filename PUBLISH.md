# 发布指南

## 发布前检查

- [ ] 更新 `clawhub.json` 中的 `homepage` 和 `support_url` 为你的 GitHub 仓库地址
- [ ] 在 `screenshots/` 目录添加 3–5 张截图（1920x1080 或 1280x720 PNG），可截取 Cursor 中 skill 评估输出界面
- [ ] 可选：录制 30–90 秒演示视频，上传到 YouTube/Vimeo，将 URL 加入 clawhub.json

## 步骤 1：发布到 GitHub

**当前状态**：skill 目录已初始化 git、完成首次 commit，remote 已指向 `https://github.com/kevin-ielts/ielts-speaking-coach.git`。

你只需完成：

1. **在 GitHub 创建仓库**（若尚未创建）  
   - 访问 https://github.com/new  
   - 仓库名：`ielts-speaking-coach`  
   - 设为 Public，不要勾选「Add a README」

2. **推送代码**（在终端执行）：
   ```bash
   cd /Users/kevin/SpeakingCoachV1/ielts-speaking-coach-skill
   git push -u origin main
   ```
   若使用 SSH：`git remote set-url origin git@github.com:kevin-ielts/ielts-speaking-coach.git` 后再 push。

若使用 GitHub CLI：`gh repo create kevin-ielts/ielts-speaking-coach --public --source=ielts-speaking-coach-skill --push`

## 步骤 2：发布到 ClawHub

```bash
# 安装 ClawHub CLI
npm i -g clawhub

# 登录（会打开浏览器）
clawhub login

# 发布 skill（使用项目根目录的副本，该目录已被 git 追踪）
clawhub publish ielts-speaking-coach-skill \
  --slug ielts-speaking-coach \
  --name "IELTS Speaking Coach" \
  --version 1.0.0 \
  --changelog "Initial release: four-criterion scoring, CHAI calibration, vocab upgrades, model answers"
```

## 权限说明（供 ClawHub 审核）

| 权限 | 用途 |
|------|------|
| network | 调用 LLM API、可选 DeepSeek 语法分析 |
| shell | 运行 generate_model_answer.py（本地模型）、load/update_trajectory 脚本 |
| all | 持久化时读写 backend/data/learner_trajectory.db |
