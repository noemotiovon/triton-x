# Triton-X

AI-driven toolkit for converting CUDA Triton Kernels to NPU (Ascend) Triton Kernels.

## Overview

Triton-X leverages AI 编程助手（支持 **GitHub Copilot**、**Cursor** 和 **Claude Code**）to automatically convert GPU Triton Kernels into NPU-compatible versions, validate correctness through accuracy testing, and continuously improve conversion quality by learning from failures.

## Features

- **Upstream Integration** — Import kernels from popular operator libraries (fla, liger-kernel, etc.)
- **Automated Conversion** — CUDA Triton → NPU Triton with rule-based guidance
- **Accuracy Testing** — Automated test generation and execution with configurable tolerances
- **Knowledge Accumulation** — Failed conversions produce structured learnings that improve future attempts

## Project Structure

```
src/
  {lib}/cuda/              GPU kernels from upstream libraries
  {lib}/ascend/upstream/   Existing NPU implementations from upstream
  {lib}/ascend/generated/  AI-generated NPU kernels
  native/cuda/             User-provided GPU kernels
  native/ascend/           AI-generated NPU kernels for user kernels
tests/                     Accuracy tests
config/                    Accuracy thresholds, upstream version tracking
docs/                      Design docs, conversion notes
```

### AI 配置目录

本项目同时为三种 AI 编程工具提供了配置，共享相同的工作流和规则：

```
.github/                   GitHub Copilot 配置
  copilot-instructions.md    全局指令
  instructions/              按文件模式匹配的上下文指令
  prompts/                   可复用的 Prompt 文件（Reusable Prompts）
.cursor/                   Cursor 配置
  rules/                     项目规则
  skills/                    AI Skills（转换、调试、测试等）
.ai/                       通用 AI Skills（Claude Code 等）
  skills/                    与 .cursor/skills/ 内容一致的 Skill 定义
CLAUDE.md                  Claude Code 项目指令
```

## Quick Start

### Prerequisites

- Python >= 3.8
- PyTorch == 2.6.0
- triton-ascend == 3.2.0
- torch_npu == 2.6.0
- CANN toolkit == 8.5.0
- pytest, PyYAML

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific library
pytest tests/fla/ -v

# Single kernel
pytest tests/fla/test_fused_recurrent_gla.py -v
```

## Usage

Triton-X 通过 AI 编程助手中的自然语言对话驱动。支持以下工具：

| 工具 | 配置目录 | 说明 |
|------|---------|------|
| **GitHub Copilot** | `.github/` | 通过 `copilot-instructions.md`、`instructions/`、`prompts/` 配置 |
| **Cursor** | `.cursor/` | 通过 `rules/`、`skills/` 配置 |
| **Claude Code** | `CLAUDE.md`, `.ai/` | 通过 `CLAUDE.md` 和 `.ai/skills/` 配置 |

三种工具共享相同的 Skills 和工作流，你可以在任一工具中使用相同的自然语言指令。

### 场景 1：集成上游算子库

将上游库（如 fla、liger-kernel）的 GPU Triton Kernel 和精度测试导入本仓库。

```
# 在 AI 助手中说：

"集成 fla 算子库，目标 commit 为 abc1234"

"把 liger-kernel 的 fused_linear_cross_entropy 算子集成进来"

"集成 fla 的所有 fused_recurrent 相关算子"
```

AI 助手会自动执行：
1. 创建 `src/fla/cuda/` 目录并复制 GPU kernel 源码
2. 检查上游是否有 NPU 实现，有则放入 `src/fla/ascend/upstream/`
3. 提取精度测试到 `tests/fla/`
4. 更新 `config/upstream.yaml` 和 `src/fla/manifest.yaml`

### 场景 2：转换 CUDA Kernel 为 NPU Kernel

#### 转换上游库中的算子

```
# 在 AI 助手中说：

"把 src/fla/cuda/fused_recurrent_gla.py 转换成 NPU 版本"

"转换 fla 库中所有 pending 状态的 kernel"
```

生成的 NPU kernel 会放在 `src/fla/ascend/generated/`。

#### 转换用户自定义算子

先将你的 CUDA kernel 放到 `src/native/cuda/` 目录下，然后：

```
# 在 AI 助手中说：

"把 src/native/cuda/my_kernel.py 转换成 NPU 版本"
```

生成的 NPU kernel 会放在 `src/native/ascend/`。

### 场景 3：运行精度测试

```
# 在 AI 助手中说：

"运行 fla 的 fused_recurrent_gla 精度测试"

"对 src/native/ascend/my_kernel.py 生成精度测试并运行"
```

也可以直接在终端运行：

```bash
# 运行全部测试
pytest tests/ -v

# 运行指定库
pytest tests/fla/ -v

# 运行单个 kernel
pytest tests/fla/test_fused_recurrent_gla.py -v

# 只运行 float16 的测试
pytest tests/fla/test_fused_recurrent_gla.py -v -k "float16"
```

### 场景 4：排查失败并沉淀经验

当精度测试失败时：

```
# 在 AI 助手中说：

"fused_recurrent_gla 的精度测试失败了，帮我排查"

"tests/fla/test_fused_recurrent_gla.py 报错 RuntimeError，帮我分析"
```

AI 助手会自动：
1. 分类失败类型（编译错误 / 运行时错误 / 精度偏差 / 测试问题）
2. 如果是测试问题 → 直接修复测试
3. 如果是 kernel 问题 → 尝试修复（最多 3 次），并将经验记录到 `docs/conversion-notes/` 和 Skills 中

### 场景 5：查看项目状态

```
# 在 AI 助手中说：

"当前 fla 库的算子转换状态是什么？"

"哪些 kernel 还没有转换？"

"列出所有 failed 状态的 kernel"
```

AI 助手会读取各 `manifest.yaml` 文件并汇总状态。

## Workflow

1. **Integrate** — Import GPU kernels and tests from upstream libraries
2. **Convert** — Generate NPU Triton Kernels from CUDA versions
3. **Test** — Run accuracy tests against reference implementations
4. **Debug & Learn** — Diagnose failures, fix issues, update Skills

See [docs/design.md](docs/design.md) for the full design document.

## Configuration

- `config/accuracy.yaml` — Tolerance thresholds per dtype and per kernel
- `config/upstream.yaml` — Upstream library version tracking

## License

TBD
