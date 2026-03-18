import yaml
import pathlib
import pytest
import torch


CONFIG_DIR = pathlib.Path(__file__).resolve().parent.parent / "config"


def load_accuracy_config():
    with open(CONFIG_DIR / "accuracy.yaml") as f:
        return yaml.safe_load(f)


_accuracy_cfg = load_accuracy_config()


def get_tolerance(dtype, kernel_name=None):
    """Return (atol, rtol) for a given dtype and optional kernel name."""
    cfg = _accuracy_cfg

    if kernel_name and kernel_name in cfg.get("kernel_overrides", {}):
        override = cfg["kernel_overrides"][kernel_name]
        return override["atol"], override["rtol"]

    dtype_name = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
    }.get(dtype)

    if dtype_name and dtype_name in cfg.get("dtype_overrides", {}):
        d = cfg["dtype_overrides"][dtype_name]
        return d["atol"], d["rtol"]

    return cfg["default"]["atol"], cfg["default"]["rtol"]


def check_accuracy(ref, out, atol, rtol, max_diff_ratio=None):
    """Unified accuracy check with detailed error reporting.

    Returns (passed: bool, report: str).
    """
    if max_diff_ratio is None:
        max_diff_ratio = _accuracy_cfg["default"].get("max_diff_ratio", 0.01)

    abs_diff = (ref.float() - out.float()).abs()
    rel_diff = abs_diff / (ref.float().abs() + 1e-12)

    max_abs = abs_diff.max().item()
    max_rel = rel_diff.max().item()
    mean_abs = abs_diff.mean().item()

    exceed_mask = (abs_diff > atol) & (rel_diff > rtol)
    exceed_ratio = exceed_mask.float().mean().item()

    passed = exceed_ratio <= max_diff_ratio
    report = (
        f"max_abs_diff={max_abs:.6e}, max_rel_diff={max_rel:.6e}, "
        f"mean_abs_diff={mean_abs:.6e}, exceed_ratio={exceed_ratio:.4%}"
    )
    return passed, report
