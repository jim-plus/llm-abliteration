import torch
from typing import Optional, Literal, Union


def sparsify_vector(
    vector: torch.Tensor,
    method: Literal["magnitude", "percentile", "topk", "soft_threshold"] = "magnitude",
    threshold: Optional[float] = None,
    **kwargs
) -> torch.Tensor:
    """
    Sparsify a vector using various methods.
    
    Args:
        vector: Input tensor to sparsify
        method: Sparsification method to use
            - "magnitude": Keep components >= threshold * max(|vector|)
            - "percentile": Keep top percentile of components by magnitude
            - "topk": Keep top k largest magnitude components
            - "soft_threshold": Apply soft thresholding (L1-like)
        threshold: Method-specific threshold parameter
        **kwargs: Additional method-specific parameters
    
    Returns:
        Sparsified vector (same shape as input)
    """
    if method == "magnitude":
        return magnitude_sparsify(vector, threshold or 0.05)
    elif method == "percentile":
        return percentile_sparsify(vector, threshold or 0.95)
    elif method == "topk":
        k = kwargs.get("k", int(0.1 * vector.numel()))
        return topk_sparsify(vector, k)
    elif method == "soft_threshold":
        return soft_threshold_sparsify(vector, threshold or 0.01)
    else:
        raise ValueError(f"Unknown method: {method}")


def magnitude_sparsify(vector: torch.Tensor, fraction: float = 0.05) -> torch.Tensor:
    """
    Keep components with magnitude >= fraction * max(|vector|).
    
    Args:
        vector: Input tensor
        fraction: Fraction of maximum magnitude (0.0 to 1.0)
    
    Returns:
        Sparsified vector
    """
    threshold = fraction * vector.abs().max()
    return torch.where(vector.abs() >= threshold, vector, torch.zeros_like(vector))


def percentile_sparsify(vector: torch.Tensor, percentile: float = 0.95) -> torch.Tensor:
    """
    Keep components above the given percentile by magnitude.
    
    Args:
        vector: Input tensor
        percentile: Percentile threshold (0.0 to 1.0). 0.95 keeps top 5%
    
    Returns:
        Sparsified vector
    """
    #threshold = vector.abs().quantile(percentile)
    threshold = torch.quantile(vector.abs().float(), percentile)
    return torch.where(vector.abs() >= threshold, vector, torch.zeros_like(vector))


def topk_sparsify(vector: torch.Tensor, k: int) -> torch.Tensor:
    """
    Keep only the top k components by magnitude.
    
    Args:
        vector: Input tensor
        k: Number of components to keep
    
    Returns:
        Sparsified vector
    """
    flat_vector = vector.view(-1)
    _, indices = torch.topk(torch.abs(flat_vector), min(k, flat_vector.numel()))
    mask = torch.zeros_like(flat_vector, dtype=torch.bool)
    mask[indices] = True
    return vector * mask.view(vector.shape)


def soft_threshold_sparsify(vector: torch.Tensor, threshold: float = 0.01) -> torch.Tensor:
    """
    Apply soft thresholding (L1 regularization-style sparsification).
    
    Args:
        vector: Input tensor
        threshold: Absolute threshold value
    
    Returns:
        Sparsified vector
    """
    return torch.sign(vector) * torch.clamp(torch.abs(vector) - threshold, min=0)


def adaptive_magnitude_sparsify(
    vector: torch.Tensor, 
    fraction: float = 0.05,
    min_components: Optional[int] = None,
    max_sparsity: float = 0.95
) -> torch.Tensor:
    """
    Enhanced magnitude-based sparsification with safeguards.
    
    Args:
        vector: Input tensor
        fraction: Fraction of maximum magnitude
        min_components: Minimum number of components to keep (prevents over-sparsification)
        max_sparsity: Maximum fraction of components to zero out
    
    Returns:
        Sparsified vector
    """
    threshold = fraction * vector.abs().max()
    
    # Count components that would be kept
    keep_mask = vector.abs() >= threshold
    components_kept = keep_mask.sum().item()
    
    # Apply minimum components safeguard
    if min_components is not None and components_kept < min_components:
        # Keep top min_components instead
        flat_vector = vector.view(-1)
        _, indices = torch.topk(torch.abs(flat_vector), min_components)
        keep_mask = torch.zeros_like(flat_vector, dtype=torch.bool)
        keep_mask[indices] = True
        keep_mask = keep_mask.view(vector.shape)
    
    # Apply maximum sparsity safeguard
    max_zeros = int(max_sparsity * vector.numel())
    if components_kept < (vector.numel() - max_zeros):
        # Keep more components to respect max_sparsity
        min_to_keep = vector.numel() - max_zeros
        flat_vector = vector.view(-1)
        _, indices = torch.topk(torch.abs(flat_vector), min_to_keep)
        keep_mask = torch.zeros_like(flat_vector, dtype=torch.bool)
        keep_mask[indices] = True
        keep_mask = keep_mask.view(vector.shape)
    
    return torch.where(keep_mask, vector, torch.zeros_like(vector))


def sparsity_stats(vector: torch.Tensor) -> dict:
    """
    Compute sparsification statistics for a vector.
    
    Args:
        vector: Input tensor
    
    Returns:
        Dictionary with sparsification statistics
    """
    total_components = vector.numel()
    nonzero_components = torch.count_nonzero(vector).item()
    sparsity = 1.0 - (nonzero_components / total_components)
    
    abs_vector = vector.abs()
    
    return {
        "total_components": total_components,
        "nonzero_components": nonzero_components,
        "sparsity": sparsity,
        "max_magnitude": abs_vector.max().item(),
        "mean_magnitude": abs_vector.mean().item(),
        "std_magnitude": abs_vector.std().item(),
        "median_magnitude": abs_vector.median().item(),
    }
