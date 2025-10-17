import torch


def magnitude_clip(vector: torch.Tensor, percentile: float = 0.99) -> torch.Tensor:
    """
    Perform symmetric magnitude Winsorization.
    Clip components to [-threshold, threshold] where threshold is at the percentile.
    
    Args:
        vector: Input tensor
        percentile: Percentile of absolute values to clip at (0.0 to 1.0)
    
    Returns:
        Clipped vector
    """

    original_dtype = vector.dtype
    vector_float = vector.float()
    abs_vector = torch.abs(vector_float)
    threshold = torch.quantile(abs_vector, percentile)
    clipped = torch.clamp(vector_float, min=-threshold, max=threshold)
    return clipped.to(original_dtype)
