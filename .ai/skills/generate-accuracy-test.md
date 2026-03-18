# Skill: Generate Accuracy Test

## Description

This skill guides the generation of accuracy tests for NPU Triton Kernels. Tests compare the NPU kernel output against a reference implementation (typically the CUDA kernel or a PyTorch-native computation).

## When to Use

- After converting a CUDA kernel to NPU and no existing test is found
- When the user explicitly asks for a test to be generated
- When an existing test is insufficient (e.g., missing dtype coverage)

## Test File Template

All accuracy tests follow this structure:

```python
"""
Accuracy test for {kernel_name}.
Source: {source_info}
"""
import pytest
import torch

# Import the reference (CUDA) implementation
# from src.{lib}.cuda.{kernel_name} import {kernel_func} as ref_kernel

# Import the NPU implementation
# from src.{lib}.ascend.generated.{kernel_name} import {kernel_func} as npu_kernel

# Import shared test utilities
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from conftest import get_tolerance, check_accuracy


# ---------------------------------------------------------------------------
# Reference implementation (pure PyTorch, runs on CPU or GPU)
# ---------------------------------------------------------------------------
def reference_impl(*args, **kwargs):
    """
    A numerically-trustworthy implementation for comparison.
    Prefer a simple PyTorch implementation over calling the CUDA Triton kernel,
    so the test does not depend on GPU hardware.
    """
    raise NotImplementedError("Fill in the reference implementation")


# ---------------------------------------------------------------------------
# NPU implementation wrapper
# ---------------------------------------------------------------------------
def npu_impl(*args, **kwargs):
    """Wrapper that invokes the NPU Triton kernel."""
    raise NotImplementedError("Fill in the NPU kernel invocation")


# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------
DTYPES = [torch.float16, torch.bfloat16, torch.float32]
SHAPES = [
    (128, 256),
    (512, 512),
    (1024, 2048),
]


# ---------------------------------------------------------------------------
# Accuracy test
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"{'x'.join(map(str, s))}")
def test_accuracy(dtype, shape):
    torch.manual_seed(42)

    # 1. Generate inputs (adapt to the kernel's actual signature)
    x = torch.randn(shape, dtype=dtype, device="cpu")

    # 2. Reference output
    ref_out = reference_impl(x)

    # 3. NPU output
    npu_out = npu_impl(x)

    # 4. Accuracy check
    atol, rtol = get_tolerance(dtype, kernel_name="{lib}/{kernel_name}")
    passed, report = check_accuracy(ref_out, npu_out, atol=atol, rtol=rtol)
    assert passed, f"Accuracy check failed: {report}"
```

## Guidelines

### Input Generation

- Use `torch.manual_seed(42)` for reproducibility
- Cover edge cases where relevant:
  - Very small / very large values
  - Zeros and negative values
  - Non-contiguous tensors (if the kernel supports them)
- For sequence-based kernels, test multiple sequence lengths

### Reference Implementation

Prefer a **pure PyTorch CPU implementation** as the reference. This avoids dependency on CUDA hardware and provides a clean numerical baseline.

If the computation is complex, the CUDA Triton kernel output (run on GPU) can serve as the reference, but note this requires GPU availability.

### Parameterization

At minimum, parameterize over:
1. **dtype**: `[float16, bfloat16, float32]`
2. **shape**: at least 3 different sizes (small, medium, large)
3. **Any kernel-specific parameters** (e.g., head_dim, num_heads)

### Accuracy Thresholds

Always use `get_tolerance()` from `conftest.py` — never hardcode thresholds. If a kernel needs special tolerances, add an override in `config/accuracy.yaml`.

### Device Placement

- Reference: run on CPU (safest numerically)
- NPU kernel: run on NPU device
- Move tensors to the correct device before calling each implementation
- Compare on CPU: `.cpu()` both results before `check_accuracy`

### Test File Placement

- Upstream library kernel → `tests/{lib}/test_{kernel_name}.py`
- Native kernel → `tests/native/test_{kernel_name}.py`

## After Generating

1. Run the test: `pytest tests/{lib}/test_{kernel_name}.py -v`
2. If it passes → update manifest status to `verified`
3. If it fails → proceed to `debug-kernel` skill
