# Triton-X 项目设计文档

## 1. 项目概述

### 1.1 项目目标

Triton-X 是一个基于 AI（Skills + Claude）驱动的工具项目，核心目标是将 GPU（CUDA）上的 Triton Kernel 自动转换为 NPU（Ascend）上可运行的 Triton Kernel，并通过自动化精度测试验证正确性。

### 1.2 核心价值

- **降低迁移成本**：自动完成 CUDA Triton → NPU Triton 的代码转换
- **知识沉淀**：将转换经验和失败教训结构化地记录在 Skills 中，持续提升转换成功率
- **生态复用**：直接集成主流算子库（liger-kernel、fla 等），快速覆盖高价值算子

### 1.3 设计原则

| 原则 | 说明 |
|------|------|
| **来源可追溯** | 每个 kernel 的来源（上游/生成/手写）、版本都可追溯 |
| **转换可复现** | 基于明确的转换规则，而非纯黑盒推理 |
| **失败可学习** | 每次失败都产出结构化经验，沉淀到 Skills |
| **渐进式扩展** | 先精度后性能，先核心算子后长尾算子 |

---

## 2. 目录结构

```
triton-x/
├── docs/                           # 项目文档
│   ├── design.md                   # 本设计文档
│   └── conversion-notes/           # 转换过程中的技术笔记
│       └── .gitkeep
│
├── src/                            # 算子源码
│   ├── {lib}/                      # 上游算子库（如 fla, liger-kernel）
│   │   ├── cuda/                   # 上游 GPU Triton Kernel（原样保留）
│   │   ├── ascend/                 # NPU Triton Kernel
│   │   │   ├── upstream/           # 上游仓库已有的 NPU 实现
│   │   │   └── generated/          # AI 生成的 NPU 实现
│   │   └── manifest.yaml           # 该库的算子清单与状态
│   │
│   └── native/                     # 用户自定义 kernel（非上游库）
│       ├── cuda/                   # 用户提供的 GPU kernel
│       ├── ascend/                 # AI 生成的 NPU kernel
│       └── manifest.yaml
│
├── tests/                          # 测试用例
│   ├── {lib}/                      # 上游库对应的测试
│   │   └── test_*.py
│   ├── native/                     # 自定义 kernel 的测试
│   │   └── test_*.py
│   └── conftest.py                 # 共享 fixtures 和精度配置
│
├── config/                         # 项目配置
│   ├── accuracy.yaml               # 精度标准配置（atol/rtol）
│   └── upstream.yaml               # 上游库版本追踪
│
├── .cursor/
│   ├── skills/                     # Cursor Skills（驱动 Claude 行为）
│   │   ├── integrate-upstream/
│   │   │   └── SKILL.md
│   │   ├── convert-cuda-to-npu/
│   │   │   └── SKILL.md
│   │   ├── generate-accuracy-test/
│   │   │   └── SKILL.md
│   │   ├── debug-kernel/
│   │   │   └── SKILL.md
│   │   └── project-conventions/
│   │       └── SKILL.md
│   └── rules/                      # Cursor Rules（项目级约定）
│       └── triton-x.md
│
└── README.md
```

### 2.1 目录约定

**`src/{lib}/` — 上游算子库**

- `cuda/`：从上游仓库原样复制的 GPU Triton Kernel，不做修改
- `ascend/upstream/`：上游仓库已有的 NPU 实现，原样复制
- `ascend/generated/`：由 AI 生成的 NPU 实现
- `manifest.yaml`：记录该库所有算子的元信息和转换状态

**`src/native/` — 用户自定义 Kernel**

- `cuda/`：用户自行编写或从其他来源获取的 GPU kernel
- `ascend/`：AI 生成的对应 NPU kernel
- `manifest.yaml`：记录状态

**`tests/`**

- 测试文件命名规则：`test_{kernel_name}.py`
- 测试文件与 kernel 一一对应
- 来源标注：上游集成的测试在文件头标注来源 commit

---

## 3. 核心工作流

### 3.1 工作流总览

```
┌─────────────────────────────────────────────────────────────────┐
│                     Triton-X 工作流                              │
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │ 1. 集成   │───▶│ 2. 转换   │───▶│ 3. 测试   │───▶│ 4. 归档   │  │
│  │ Integrate │    │ Convert  │    │ Test     │    │ Archive  │  │
│  └──────────┘    └──────────┘    └──────────┘    └─────┬────┘  │
│                                       │                 │       │
│                                       │ 失败             │       │
│                                       ▼                 │       │
│                                  ┌──────────┐           │       │
│                                  │ 5. 诊断   │───────────┘       │
│                                  │ Debug    │  修复后重试         │
│                                  └──────────┘                    │
│                                       │                          │
│                                       ▼                          │
│                                  ┌──────────┐                    │
│                                  │ 6. 沉淀   │                    │
│                                  │ Learn    │                    │
│                                  └──────────┘                    │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 阶段详述

#### 阶段 1：集成上游算子库（Integrate）

**触发条件**：用户指定要集成某个上游库（如 `fla`）

**执行步骤**：
1. 确认上游仓库地址和目标 commit/版本
2. 提取目标算子的 GPU Triton Kernel 源码 → `src/{lib}/cuda/`
3. 检查上游是否已有 NPU 实现 → 如有，放入 `src/{lib}/ascend/upstream/`
4. 提取对应的精度测试用例 → `tests/{lib}/`
5. 记录版本信息 → `config/upstream.yaml`
6. 更新算子清单 → `src/{lib}/manifest.yaml`

**对应 Skill**：`integrate-upstream`

#### 阶段 2：CUDA → NPU 转换（Convert）

**触发条件**：用户指定要转换某个 kernel

**执行步骤**：
1. 读取目标 CUDA Triton Kernel
2. 加载转换规则 Skills（`convert-cuda-to-npu`）
3. 分析 kernel 使用的特性（内存操作、并行模式、特殊函数等）
4. 根据规则生成 NPU Triton Kernel
5. 写入对应目录：
   - 上游库算子 → `src/{lib}/ascend/generated/`
   - 自定义算子 → `src/native/ascend/`
6. 更新 `manifest.yaml` 状态为 `converted`

**对应 Skill**：`convert-cuda-to-npu`

#### 阶段 3：精度测试（Test）

**触发条件**：转换完成后自动执行，或用户手动触发

**执行步骤**：
1. 检查是否存在对应的精度测试
   - 存在 → 直接运行
   - 不存在 → 调用 `generate-accuracy-test` Skill 生成
2. 运行精度测试
3. 根据 `config/accuracy.yaml` 中的标准判定通过/失败
4. 更新 `manifest.yaml` 状态

**对应 Skill**：`generate-accuracy-test`

#### 阶段 4：归档（Archive）

**触发条件**：精度测试通过

**执行步骤**：
1. 更新 `manifest.yaml` 状态为 `verified`
2. 记录测试通过时的环境信息（torch 版本、CANN 版本等）

#### 阶段 5：诊断（Debug）

**触发条件**：精度测试失败

**执行步骤**：
1. 分析失败类型：
   - **编译错误**：kernel 代码语法或 API 问题
   - **运行时错误**：执行过程中 crash
   - **精度不达标**：结果偏差超出阈值
   - **测试用例问题**：测试代码本身有误（如输入构造错误、设备放置遗漏）
2. 根据类型分别处理：
   - 测试用例问题 → 自动修复测试代码 → 重新运行（阶段 3）
   - kernel 问题 → 记录详细错误 → 进入阶段 6
3. 重试机制：
   - 每个 kernel 最多自动重试 **3 次**
   - 每次重试前检查是否有新的 Skills 知识可用
   - 超过重试次数 → 标记为 `failed`，等待人工介入

**对应 Skill**：`debug-kernel`

#### 阶段 6：知识沉淀（Learn）

**触发条件**：kernel 问题导致转换失败

**执行步骤**：
1. 提取关键错误信息（错误类型、涉及的 API、错误消息）
2. 搜索相关资料（NPU Triton 文档、已知限制、社区 issue）
3. 总结为结构化经验，更新到对应 Skill 中
4. 记录到 `docs/conversion-notes/` 作为技术笔记

---

## 4. 配置文件设计

### 4.1 精度标准 — `config/accuracy.yaml`

```yaml
default:
  atol: 1e-4
  rtol: 1e-4
  max_diff_ratio: 0.01      # 允许最多 1% 的元素超出阈值

dtype_overrides:
  float16:
    atol: 1e-3
    rtol: 1e-3
  bfloat16:
    atol: 1e-2
    rtol: 1e-2
  float32:
    atol: 1e-5
    rtol: 1e-5

kernel_overrides: {}         # 特定 kernel 的自定义精度标准
```

### 4.2 上游版本追踪 — `config/upstream.yaml`

```yaml
libraries:
  fla:
    repo: "https://github.com/sustcsonglin/flash-linear-attention"
    commit: ""               # 集成时填入具体 commit hash
    version: ""              # 如有 release tag
    integrated_at: ""        # 集成日期
    kernels: []              # 集成的 kernel 列表

  liger-kernel:
    repo: "https://github.com/linkedin/Liger-Kernel"
    commit: ""
    version: ""
    integrated_at: ""
    kernels: []
```

### 4.3 算子清单 — `src/{lib}/manifest.yaml`

```yaml
library: "fla"
source_commit: "abc1234"

kernels:
  - name: "fused_recurrent_gla"
    cuda_path: "cuda/fused_recurrent_gla.py"
    ascend_path: "ascend/generated/fused_recurrent_gla.py"
    test_path: "tests/fla/test_fused_recurrent_gla.py"
    source: "generated"        # upstream | generated | manual
    status: "verified"         # pending | converted | testing | verified | failed
    last_updated: "2026-03-17"
    notes: ""

  - name: "chunk_gla"
    cuda_path: "cuda/chunk_gla.py"
    ascend_path: ""
    test_path: ""
    source: ""
    status: "pending"
    last_updated: ""
    notes: ""
```

---

## 5. Skills 体系设计

Skills 是驱动 Claude 完成各阶段任务的核心知识库。每个 Skill 对应一个明确的职责。

### 5.1 Skills 总览

| Skill | 路径 | 职责 |
|-------|------|------|
| `project-conventions` | `.cursor/skills/project-conventions/SKILL.md` | 项目约定：目录结构、命名规范、文件组织 |
| `integrate-upstream` | `.cursor/skills/integrate-upstream/SKILL.md` | 集成上游算子库的标准流程 |
| `convert-cuda-to-npu` | `.cursor/skills/convert-cuda-to-npu/SKILL.md` | CUDA → NPU 的转换规则和已知 pattern |
| `generate-accuracy-test` | `.cursor/skills/generate-accuracy-test/SKILL.md` | 精度测试生成规范 |
| `debug-kernel` | `.cursor/skills/debug-kernel/SKILL.md` | 失败诊断流程和经验积累 |

### 5.2 Skills 演进机制

Skills 不是静态文档，而是**持续演进的知识库**：

```
转换失败 → 分析原因 → 提取 pattern → 更新 Skill → 下次转换受益
```

**更新规则**：
- 每次遇到新的转换 pattern，追加到 `convert-cuda-to-npu` 的规则列表
- 每次遇到新的失败类型，追加到 `debug-kernel` 的经验库
- 定期 review Skills 内容，合并重复项，优化结构

### 5.3 各 Skill 核心内容概要

#### `project-conventions`
- 目录结构约定（如本文档第 2 节）
- 文件命名规范
- manifest.yaml 维护规则
- commit message 规范

#### `integrate-upstream`
- 如何从上游仓库定位目标算子
- 文件复制和组织流程
- 测试用例的提取与适配
- upstream.yaml 更新流程

#### `convert-cuda-to-npu`
- CUDA Triton 与 NPU Triton 的 API 差异映射表
- 内存操作转换规则（tl.load/tl.store 等）
- 并行模式映射（program_id、grid 等）
- 常见转换 pattern 与示例
- 已知限制清单（NPU Triton 不支持的特性）

#### `generate-accuracy-test`
- 测试文件模板
- 输入数据生成策略
- 精度比对方法（参考 config/accuracy.yaml）
- 多 dtype 测试覆盖要求
- 设备放置（device placement）注意事项

#### `debug-kernel`
- 失败分类标准（编译错误 / 运行时错误 / 精度偏差 / 测试问题）
- 各类失败的排查 checklist
- 历史失败案例与解决方案（持续积累）
- 重试策略与上限

---

## 6. 精度测试规范

### 6.1 测试结构

每个测试文件遵循统一模式：

```python
import pytest
import torch
import triton

def reference_impl(...):
    """CPU/GPU 参考实现，用于对比"""
    pass

def npu_impl(...):
    """调用 NPU Triton Kernel"""
    pass

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(128, 256), (512, 512), (1024, 2048)])
def test_accuracy(dtype, shape):
    """精度对比测试"""
    # 1. 构造输入
    # 2. 运行参考实现
    # 3. 运行 NPU 实现
    # 4. 精度对比
    pass
```

### 6.2 精度对比方法

```python
def check_accuracy(ref, out, atol, rtol, max_diff_ratio=0.01):
    """
    统一精度检查：
    1. torch.allclose 整体检查
    2. 逐元素检查超出阈值的比例
    3. 最大绝对/相对误差报告
    """
    pass
```

### 6.3 测试运行

```bash
# 运行指定库的所有精度测试
pytest tests/fla/ -v

# 运行单个 kernel 测试
pytest tests/fla/test_fused_recurrent_gla.py -v

# 运行所有测试
pytest tests/ -v
```

---

## 7. 环境要求

### 7.1 基础依赖

| 依赖 | 说明 |
|------|------|
| Python | >= 3.8 |
| PyTorch | >= 2.1 |
| Triton | GPU 版本（用于参考实现） |
| triton-ascend | NPU 版本的 Triton |
| torch_npu | PyTorch NPU 适配层 |
| CANN | 华为 Ascend 计算框架 |
| pytest | 测试框架 |
| PyYAML | 配置文件解析 |

### 7.2 环境验证

项目应提供一个环境检查脚本，验证所有依赖是否正确安装：

```bash
python scripts/check_env.py
```

---

## 8. 后续扩展规划

### Phase 1（当前）— 精度验证
- 集成核心算子库
- 完成 CUDA → NPU 转换流程
- 精度测试覆盖

### Phase 2 — 性能测试
- 增加 benchmark 测试框架
- 性能对比（NPU vs GPU）
- 性能优化 Skills 积累

### Phase 3 — 自动化流水线
- CI/CD 集成
- 批量转换与测试
- 转换成功率统计与报告

### Phase 4 — 社区与生态
- 支持更多上游算子库
- 转换规则社区共建
- 可能支持更多 NPU 平台
