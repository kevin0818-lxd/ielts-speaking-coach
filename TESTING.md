# IELTS Speaking Coach — 测试说明

## 快速验证

### 1. 脚本可执行性

```bash
# 从项目根目录执行
cd /path/to/SpeakingCoachV1

# generate_model_answer.py（需 mlx-lm，无则报错退出）
python .cursor/skills/ielts-speaking-coach/scripts/generate_model_answer.py \
  --transcript "I am a student. I study business." --part 1 --user-band 6.0 --json

# load_trajectory.py（需 backend 可导入，否则返回空 trajectory_targets）
ONTOLOGY_TRAJECTORY_ENABLED=1 python .cursor/skills/ielts-speaking-coach/scripts/load_trajectory.py --user-id default

# update_trajectory.py（需 backend，否则静默失败）
echo '{"text":"...","part":1,"breakdown":{...},"recommendations":[...]}' | \
  ONTOLOGY_TRAJECTORY_ENABLED=1 python .cursor/skills/ielts-speaking-coach/scripts/update_trajectory.py --user-id default --part 1
```

### 2. 依赖要求

| 脚本 | 必需依赖 | 无依赖时行为 |
|------|----------|--------------|
| generate_model_answer.py | mlx-lm | 退出码 1，提示安装 |
| load_trajectory.py | backend (fastapi, torch, etc.) | 返回 `{"trajectory_targets":[], "error":"..."}` |
| update_trajectory.py | backend | 静默退出或超时 |

### 3. 完整环境测试

使用项目 venv（含 mlx、backend 依赖）：

```bash
.venv-mlx/bin/python .cursor/skills/ielts-speaking-coach/scripts/generate_model_answer.py ...
```

持久化脚本需 `ONTOLOGY_TRAJECTORY_ENABLED=1` 且 backend 可导入。

## 已知限制

- **generate_model_answer.py**：仅支持 Apple Silicon（MLX）；Windows/Linux 需其他实现
- **load/update_trajectory**：依赖 backend 完整环境，沙盒或精简环境可能导入失败
- **Pronunciation**：文本输入时 PR 为 FC/LR/GRA 中位数估计，非真实发音评分
