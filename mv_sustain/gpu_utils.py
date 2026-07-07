"""GPU device management for SuStaIn likelihood kernels.

Enable GPU via environment variables before launching sustain_main.py:

    SUSTAIN_USE_GPU=1                   # required to activate GPU path
    SUSTAIN_GPU_DEVICE=0                # which CUDA device index (default 0)

When SUSTAIN_USE_GPU is not set (or 0), get_torch_device() returns (None, None)
and callers fall back to the existing numpy CPU path with no overhead.
"""

import os
import warnings

_cached: tuple | None = None  # (torch_module, torch.device) | (None, None)


def get_torch_device() -> tuple:
    """Return (torch, device) if GPU is available and enabled, else (None, None).

    Result is cached after the first call — subsequent calls are essentially free.
    """
    global _cached
    if _cached is not None:
        return _cached

    if os.environ.get("SUSTAIN_USE_GPU", "0") != "1":
        _cached = (None, None)
        return _cached

    try:
        import torch  # noqa: PLC0415
    except ImportError:
        warnings.warn(
            "SUSTAIN_USE_GPU=1 but PyTorch is not installed; "
            "falling back to CPU numpy path.",
            stacklevel=2,
        )
        _cached = (None, None)
        return _cached

    gpu_idx = int(os.environ.get("SUSTAIN_GPU_DEVICE", "0"))
    if not torch.cuda.is_available():
        warnings.warn(
            "SUSTAIN_USE_GPU=1 but no CUDA device found; "
            "falling back to CPU numpy path.",
            stacklevel=2,
        )
        _cached = (None, None)
        return _cached

    device = torch.device(f"cuda:{gpu_idx}")
    _cached = (torch, device)
    return _cached
