# Skill: Debug Kernel

## Description

This skill defines the systematic approach for diagnosing and resolving failures when an NPU Triton Kernel does not pass accuracy tests. It also serves as a living knowledge base of encountered issues and their solutions.

## When to Use

- Accuracy test fails after kernel conversion
- Kernel compilation errors on NPU
- Runtime errors during kernel execution
- Any unexpected behavior in converted kernels

## Diagnosis Process

### Step 1: Classify the Failure

Read the error output and classify into one of these categories:

| Category | Indicators | Priority |
|----------|-----------|----------|
| **Compilation Error** | `CompilationError`, `SyntaxError`, triton compiler messages | Fix kernel code |
| **Runtime Error** | `RuntimeError`, segfault, device errors | Fix kernel or launch config |
| **Accuracy Failure** | Test assertion fails, `check_accuracy` reports high diff | Fix kernel logic |
| **Test Issue** | `ImportError`, wrong shapes, device mismatch in test | Fix test code |

### Step 2: Handle by Category

#### A. Test Issue (fix the test)

Common test problems:
1. **Import path errors** — verify imports match our directory structure
2. **Device placement** — ensure input tensors are on the correct device (NPU)
3. **Shape mismatch** — verify test generates inputs matching the kernel's expectations
4. **Missing dependencies** — check if test needs packages not yet installed

**Action**: Fix the test code directly, then re-run. This does NOT count toward the kernel retry limit.

#### B. Compilation Error (fix the kernel)

Checklist:
1. Check for unsupported Triton operations on NPU
2. Check data type compatibility
3. Check `tl.constexpr` block size constraints
4. Verify pointer arithmetic correctness
5. Check for CUDA-specific intrinsics that have no NPU equivalent

**Action**: Fix the kernel, increment retry count, re-run test.

#### C. Runtime Error (fix the kernel or launch config)

Checklist:
1. Verify grid/block dimensions are valid for NPU
2. Check for out-of-bounds memory access
3. Verify tensor strides and memory layout
4. Check for race conditions in concurrent writes

**Action**: Fix the kernel or launch config, increment retry count, re-run test.

#### D. Accuracy Failure (fix the kernel logic)

Checklist:
1. Compare intermediate values (insert debug prints if possible)
2. Test with float32 only — if float32 passes but float16 fails, it is likely a precision issue, not a logic bug
3. Check reduction operation ordering (NPU may accumulate differently)
4. Verify mask logic for boundary handling
5. Check if the reference implementation is correct

**Action**: Fix the kernel, increment retry count, re-run test.

### Step 3: Retry or Escalate

```
Retry count < 3?
├── Yes → Apply fix, go back to Step 1
└── No  → Mark as "failed", proceed to Knowledge Capture
```

### Step 4: Knowledge Capture

When a failure is resolved OR when max retries are exhausted:

1. **Record the issue** in `docs/conversion-notes/{kernel_name}.md`:
   ```markdown
   # {kernel_name} Conversion Notes

   ## Issue
   {Brief description}

   ## Error Message
   {Key error output}

   ## Root Cause
   {Analysis}

   ## Solution
   {What fixed it, or "Unresolved — needs manual investigation"}

   ## Takeaway
   {General lesson that applies to future conversions}
   ```

2. **Update Skills** if the issue reveals a general pattern:
   - New conversion rule → update `convert-cuda-to-npu` Conversion Rule Table
   - New known limitation → update `convert-cuda-to-npu` Known Limitations
   - New test pattern → update `generate-accuracy-test` Guidelines

---

## Known Issues Library

> This section accumulates resolved issues as a searchable reference.
> Format: `[Category] Brief description → Solution`

_(To be populated as issues are encountered)_

### Compilation Issues

<!-- Example:
- [Compilation] `tl.atomic_add` not supported on NPU → Use `tl.store` with explicit reduction
-->

### Runtime Issues

<!-- Example:
- [Runtime] Block size 1024 exceeds NPU limit → Reduce to 512 or smaller
-->

### Accuracy Issues

<!-- Example:
- [Accuracy] float16 softmax diverges for large sequences → Use float32 accumulator
-->

### Test Issues

<!-- Example:
- [Test] Missing `torch_npu` import → Add conditional import at top of test
-->
