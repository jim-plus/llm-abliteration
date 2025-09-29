import gc
import torch
from tqdm import tqdm
from transformers import PreTrainedModel


def modify_tensor(
    tensor_data: torch.Tensor, refusal_dir: torch.Tensor, scale_factor: float = 1.0
) -> torch.nn.Parameter:
    if tensor_data.device != refusal_dir.device:
        refusal_dir = refusal_dir.to(tensor_data.device)
    tensor_float32 = tensor_data.to(torch.float32)
    refusal_dir_float32 = refusal_dir.to(torch.float32)
    # Ensure refusal_dir is a 1-dimensional tensor
    if refusal_dir_float32.dim() > 1:
        refusal_dir_float32 = refusal_dir_float32.view(-1)
    tensor_float32 -= scale_factor * torch.matmul(
        torch.outer(refusal_dir_float32, refusal_dir_float32), tensor_float32
    )
    tensor_modified = tensor_float32.to(torch.bfloat16)

    torch.cuda.empty_cache()
    gc.collect()

    return torch.nn.Parameter(tensor_modified)


def modify_tensor_refactored(
    tensor_data: torch.Tensor, refusal_dir: torch.Tensor, scale_factor: float = 1.0
) -> torch.nn.Parameter:
    if tensor_data.device != refusal_dir.device:
        refusal_dir = refusal_dir.to(tensor_data.device)

    # Use inplace operations to minimize memory allocation
    tensor_float32 = tensor_data.to(torch.float32)
    refusal_dir_float32 = refusal_dir.to(torch.float32)

    if refusal_dir_float32.dim() > 1:
        refusal_dir_float32 = refusal_dir_float32.view(-1)

    # Optimized computation: instead of computing outer product then matrix multiply,
    # use the identity (v ⊗ v) @ A = v @ (v^T @ A) where ⊗ is outer product
    # This avoids creating the full matrix
    projection = torch.matmul(refusal_dir_float32, tensor_float32)
    tensor_float32 -= scale_factor * torch.outer(refusal_dir_float32, projection)

    # Return original tensor parameter as required
    tensor_modified = tensor_float32.to(tensor_data.dtype)

    torch.cuda.empty_cache()
    gc.collect()

    return torch.nn.Parameter(tensor_modified)


def modify_tensor_improved(
    tensor_data: torch.Tensor, refusal_dir: torch.Tensor, scale_factor: float = 1.0
) -> torch.nn.Parameter:
    original_device = tensor_data.device
    original_dtype = tensor_data.dtype

    # Move tensors to GPU for computation
    tensor_gpu = tensor_data.to('cuda', dtype=torch.float32, non_blocking=True)
    refusal_dir_gpu = refusal_dir.to('cuda', dtype=torch.float32, non_blocking=True)

    # Ensure refusal_dir is a 1-dimensional tensor
    if refusal_dir_gpu.dim() > 1:
        refusal_dir_gpu = refusal_dir_gpu.view(-1)

    # Optimized computation: instead of computing outer product then matrix multiply,
    # use the identity (v ⊗ v) @ A = v @ (v^T @ A) where ⊗ is outer product
    # This avoids creating the full [4096, 4096] matrix
    projection = torch.matmul(refusal_dir_gpu, tensor_gpu)  # Shape: [14336]
    tensor_gpu -= scale_factor * torch.outer(refusal_dir_gpu, projection)

    # Convert back to original dtype and move back to original device
    tensor_modified = tensor_gpu.to(original_device, dtype=original_dtype, non_blocking=True)
    torch.cuda.empty_cache()
    gc.collect()
    return torch.nn.Parameter(tensor_modified)


def apply_abliteration(
    model: PreTrainedModel,
    refusal_dir: torch.Tensor,
    skip_begin_layers: int = 1,
    skip_end_layers: int = 0,
    scale_factor: float = 1.0,
) -> PreTrainedModel:
    refusal_dir = torch.nn.functional.normalize(refusal_dir, dim=-1)
    lm_model = model.model
    assert hasattr(
        lm_model, "layers"
    ), "The model does not have the expected structure."
    num_layers = len(lm_model.layers)
    for layer_idx in tqdm(
        range(skip_begin_layers, num_layers - skip_end_layers),
        desc="Applying abliteration",
    ):
        lm_model.layers[layer_idx].self_attn.o_proj.weight = modify_tensor_improved(
            lm_model.layers[layer_idx].self_attn.o_proj.weight.data,
            refusal_dir,
            scale_factor,
        )
        lm_model.layers[layer_idx].mlp.down_proj.weight = modify_tensor_improved(
            lm_model.layers[layer_idx].mlp.down_proj.weight.data,
            refusal_dir,
            scale_factor,
        )

    torch.cuda.empty_cache()
    gc.collect()

    return model
