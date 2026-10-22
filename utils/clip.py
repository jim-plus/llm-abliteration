import torch


def magnitude_clip(vector: torch.Tensor, percentile: float = 0.99) -> torch.Tensor:
    """
    Perform symmetric magnitude Winsorization.
    Clip components to [-threshold, threshold] where threshold is at the percentile.
    Compute the threshold along the last dimension independently for each vector,
    so batching prompts does not change their clipping thresholds.
    
    Args:
        vector: Input tensor
        percentile: Percentile of absolute values to clip at (0.0 to 1.0)
    
    Returns:
        Clipped vector
    """

    original_dtype = vector.dtype
    vector_d = vector.double()
    abs_vector = torch.abs(vector_d)
    threshold = torch.quantile(abs_vector, percentile, dim=-1, keepdim=True)
    clipped = torch.clamp(vector_d, min=-threshold, max=threshold)
    return clipped.to(original_dtype)
