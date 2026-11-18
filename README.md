# DexJoCo 触觉正式评测

本仓库整理了 [tactile](https://github.com/HERMES-Unifyed-Robotic-Control/tactile) 固定版本
`b5bdc08` 的触觉接入代码，以及普通晚融合和轻量闭环的正式评测入口。

评测客户端导入 [client_touch_patch.py](single_stage_touch/code/sim_touch/client_touch_patch.py)，
由 [touch_hook.py](single_stage_touch/code/sim_touch/touch_hook.py) 读取物理子步中的指尖接触力。
客户端按控制步把触觉写入 `touch` 和 `touch_mask`，与该步状态一起交给策略。
原始项目的 [run_with_touch.py](single_stage_touch/code/sim_touch/run_with_touch.py) 可为兼容的
DexJoCo 客户端安装同一补丁。

| 评测 | 入口 | 控制时序 |
|---|---|---|
| 普通触觉晚融合 | [run_formal.py](single_stage_late_touch/eval4090/run_formal.py) | H30，异步重规划阈值 0.8，每个请求读取当前触觉 |
| 轻量闭环 | [mission.py](single_stage_closed_loop/eval4090/mission.py) | E/锚点保持 30 步，每 5 步采样状态与触觉并修正下一块 |

两套正式入口均使用 `rand_obj`、seed 0 和 11 个任务各 50 局。普通触觉可另用 `mask0`
运行同一模型的触觉屏蔽对照。轻量版本使用同步五动作客户端和执行确认。

## 部署前提

沿用已准备的 Linux CUDA/EGL 评测部署。设置 `AFCE_EVAL_ROOT` 指向部署根目录，
并用部署的环境脚本配置 MuJoCo、OpenPI、tactile 及模拟器的导入路径。评测根目录需要：

- `repo/` 和 `closed_loop_eval/repo/` 对应本仓库源码。
- 普通触觉的 `runtime_private/eval_harness/` 使用已固定版本的上游评测器与审计服务；
  轻量版的 `closed_loop_eval/runtime_private_v2/` 提供策略运行时。
- `venv/bin/python` 为策略解释器；`layout/sim/simenv/bin/python` 为仿真解释器。
  仿真侧需要官方 `8d23b0f` 配置与 `layout/sim/verify_video.py`。
- `rebuilt_native_v1/datasets/` 提供与策略一致的数据元信息，
  `OPENPI_DATA_HOME` 指向 tokenizer 缓存。
- 普通触觉权重位于 `models/single_stage_late_touch_touch_seed42/`，须有 60000 更新的
  `MIGRATION_VERIFIED.json`；轻量权重位于 `closed_loop_eval/light_seed42/`，
  须有对应的 `closed_loop_eval/LIGHT_MIGRATION_VERIFIED.json`。评测脚本会检查模型身份。

分支包含评测源码。策略资产、独立运行时和驱动环境由执行方提供。

## 运行

普通触觉使用部署的普通触觉环境脚本。先验证固定评测器源码与替换点，再做加载检查：

```bash
source "$AFCE_EVAL_ROOT/env.sh"
policy_python="$AFCE_EVAL_ROOT/venv/bin/python"
static="$AFCE_EVAL_ROOT/repo/single_stage_late_touch/eval4090"
"$policy_python" "$static/evaluate.py" --check-patches
"$policy_python" "$static/smoke_final.py" --gpu 0
"$policy_python" "$static/run_formal.py" --gpu 0 --condition real --seed 0
```

轻量闭环使用其独立环境脚本。环境的 `PYTHONPATH` 还需包含普通评测目录（供
`observation_adapter`）、`single_stage_touch/code/sim_touch` 和固定模拟器的 Python 根目录。
轻量调度入口读取普通触觉正式 550 局的 `results/real/seed0/complete.json`，
随后在空闲 4090 上执行单臂和双臂检查以及全量评测：

```bash
source "$AFCE_EVAL_ROOT/closed_loop_eval/env.sh"
policy_python="$AFCE_EVAL_ROOT/venv/bin/python"
light="$AFCE_EVAL_ROOT/closed_loop_eval/repo/single_stage_closed_loop/eval4090"
"$policy_python" "$light/mission.py"
```

自动普通触觉多卡调度可用同目录的 `mission.py`。已经完成的评测目录包含逐局回执、
视频、续跑边界或反馈轨迹。汇总结果以各自输出根目录的 `complete.json` 与
`REPORT.md` 为准，检查局单独计数。
