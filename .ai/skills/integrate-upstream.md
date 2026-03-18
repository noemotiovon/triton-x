# Skill: Integrate Upstream Library

## Description

This skill guides the process of integrating GPU Triton Kernels and their tests from upstream operator libraries (e.g., fla, liger-kernel) into the Triton-X repository.

## When to Use

When the user asks to integrate kernels from an upstream library, such as:
- "集成 fla 的算子"
- "把 liger-kernel 的 xxx 算子加进来"
- "Add fla's fused_recurrent kernel"

## Prerequisites

Before integrating, confirm with the user:
1. **Which library** to integrate (e.g., `fla`)
2. **Which kernels** to integrate (specific ones, or all available)
3. **Target version/commit** of the upstream repo (default: latest main)

## Step-by-Step Process

### Step 1: Prepare Directory Structure

```bash
mkdir -p src/{lib}/cuda
mkdir -p src/{lib}/ascend/upstream
mkdir -p src/{lib}/ascend/generated
mkdir -p tests/{lib}
```

### Step 2: Identify and Extract GPU Kernels

1. Browse the upstream repository to locate Triton kernel files
2. Common locations to check:
   - `{lib}/ops/triton/`
   - `{lib}/kernels/`
   - `{lib}/triton/`
3. Copy the GPU Triton kernel source files to `src/{lib}/cuda/`
4. Add source header to each file (repo URL + commit hash)

### Step 3: Check for Existing NPU Implementations

1. Search the upstream repo for Ascend/NPU implementations
2. Common indicators: `npu`, `ascend`, `torch_npu` in file paths or imports
3. If found, copy to `src/{lib}/ascend/upstream/`

### Step 4: Extract Accuracy Tests

1. Locate test files in the upstream repo (usually `tests/` or `test/`)
2. Focus on **accuracy/correctness tests** — skip performance benchmarks
3. Copy relevant test files to `tests/{lib}/`
4. Adapt tests if needed:
   - Update import paths to match our directory structure
   - Ensure tests can run independently
   - Use our `conftest.py` utilities where appropriate

### Step 5: Update Configuration

Update `config/upstream.yaml`:
```yaml
libraries:
  {lib}:
    repo: "{repo_url}"
    commit: "{commit_hash}"
    version: "{version_tag_if_any}"
    integrated_at: "{today's date}"
    kernels:
      - "kernel_name_1"
      - "kernel_name_2"
```

### Step 6: Create Manifest

Create `src/{lib}/manifest.yaml` with entries for each kernel:
```yaml
library: "{lib}"
source_commit: "{commit_hash}"

kernels:
  - name: "{kernel_name}"
    cuda_path: "cuda/{kernel_name}.py"
    ascend_path: ""
    test_path: "tests/{lib}/test_{kernel_name}.py"
    source: ""
    status: "pending"
    last_updated: "{today's date}"
    notes: ""
```

### Step 7: Verify Integration

1. Confirm all files are in the correct locations
2. Run a quick import check to ensure no missing dependencies
3. If GPU environment is available, optionally run CUDA tests to verify they pass on GPU

## Checklist

- [ ] Kernel files copied to `src/{lib}/cuda/`
- [ ] Upstream NPU implementations (if any) copied to `src/{lib}/ascend/upstream/`
- [ ] Accuracy tests copied and adapted in `tests/{lib}/`
- [ ] `config/upstream.yaml` updated with version info
- [ ] `src/{lib}/manifest.yaml` created
- [ ] File headers added with source attribution
