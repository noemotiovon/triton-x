# Triton-X Project Instructions

## Project Context

Triton-X converts CUDA Triton Kernels to NPU (Ascend) Triton Kernels using AI-driven Skills.

## Skills

Before performing any of the following tasks, **always read the corresponding Skill file first**:

| Task | Skill File |
|------|-----------|
| Working in this repo (always) | `.ai/skills/project-conventions.md` |
| Integrating an upstream library | `.ai/skills/integrate-upstream.md` |
| Converting a CUDA kernel to NPU | `.ai/skills/convert-cuda-to-npu.md` |
| Generating an accuracy test | `.ai/skills/generate-accuracy-test.md` |
| Debugging a failed kernel/test | `.ai/skills/debug-kernel.md` |

## Mandatory Behaviors

1. **Always update `manifest.yaml`** after adding, converting, or testing a kernel.
2. **Never modify files under `cuda/` or `ascend/upstream/`** — these are upstream snapshots.
3. **Use `config/accuracy.yaml` thresholds** — never hardcode tolerance values in tests.
4. **Add file headers** to all generated and upstream-sourced files.
5. **Record conversion failures** in `docs/conversion-notes/` and update Skills.
6. **Maximum 3 automatic retries** for kernel conversion failures.
