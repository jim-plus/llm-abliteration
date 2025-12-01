import torch


def get_preferred_device(requested: str | None = "auto") -> str:
    """
    Choose a runtime device based on user request and availability.
    Falls back in priority order: CUDA -> MPS -> CPU.
    """
    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_device_map(requested: str | None = "auto") -> str:
    """
    Preserve HuggingFace's automatic sharding when 'auto' is requested,
    otherwise return the concrete device string.
    """
    return "auto" if requested in (None, "auto") else requested


def clear_device_cache() -> None:
    """
    Release cached memory for the active accelerator, if any.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def synchronize_device(device: str | None = None) -> None:
    """
    Block until all queued kernels on the accelerator complete.
    """
    target = device
    if target is None or target == "auto":
        if torch.cuda.is_available():
            target = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            target = "mps"
        else:
            target = "cpu"

    if target == "cuda":
        torch.cuda.synchronize()
    elif target == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()
