import gc
import torch
from tqdm import tqdm
from transformers import PreTrainedModel
from utils.sparsify import magnitude_sparsify, sparsity_stats


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

    del tensor_float32
    del refusal_dir_float32

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

    del refusal_dir_float32
    del projection

    # Return original tensor parameter as required
    tensor_modified = tensor_float32.to(tensor_data.dtype)

    del tensor_float32

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
    # This avoids creating the full matrix
    projection = torch.matmul(refusal_dir_gpu, tensor_gpu)
    tensor_gpu -= scale_factor * torch.outer(refusal_dir_gpu, projection)

    del refusal_dir_gpu
    del projection

    # Convert back to original dtype and move back to original device
    tensor_modified = tensor_gpu.to(original_device, dtype=original_dtype, non_blocking=True)
    del tensor_gpu
    torch.cuda.empty_cache()
    gc.collect()
    return torch.nn.Parameter(tensor_modified)

# more aggressive about returning VRAM allocation
def modify_tensor_improved2(
    tensor_data: torch.Tensor, refusal_dir: torch.Tensor, scale_factor: float = 1.0,
) -> torch.nn.Parameter:
    original_device = tensor_data.device
    original_dtype = tensor_data.dtype

    # Use no_grad context to prevent graph construction
    with torch.no_grad():
        # Move tensors to GPU for computation
        tensor_gpu = tensor_data.to('cuda', dtype=torch.float32, non_blocking=True)
        refusal_dir_gpu = refusal_dir.to('cuda', dtype=torch.float32, non_blocking=True)

        # Ensure refusal_dir is a 1-dimensional tensor
        if refusal_dir_gpu.dim() > 1:
            refusal_dir_gpu = refusal_dir_gpu.view(-1)

        # Optimized computation
        projection = torch.matmul(refusal_dir_gpu, tensor_gpu)
        tensor_gpu -= scale_factor * torch.outer(refusal_dir_gpu, projection)

        del projection
        del refusal_dir_gpu
        
        # Convert back to original dtype and move back to original device
        tensor_modified = tensor_gpu.to(original_device, dtype=original_dtype, non_blocking=True)

        del tensor_gpu
        
        # Force synchronization and clear cache
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # Create a fresh tensor with no computation history
    # .detach() ensures no gradient tracking, .clone() breaks all references
    clean_tensor = tensor_modified.detach().clone()
    del tensor_modified
    
    # Create Parameter from the clean tensor
    return torch.nn.Parameter(clean_tensor)  # Defaults to requires_grad=True


def apply_abliteration(
    model: PreTrainedModel,
    refusal_dir: torch.Tensor,
    layer_target: int,
    skip_begin_layers: int = 1,
    skip_end_layers: int = 0,
    scaling: float = 1.0,
    sparsity: float = 0.0,
) -> PreTrainedModel:
    # sparsify before normalizing!
    if sparsity > 0.0:
        print(sparsity_stats(refusal_dir))
        refusal_dir = magnitude_sparsify(refusal_dir, fraction=sparsity)
        print(sparsity_stats(refusal_dir))

    refusal_dir = torch.nn.functional.normalize(refusal_dir, dim=-1)
    layer_base = model.model
    if hasattr(layer_base,"language_model"):
        layer_base = layer_base.language_model
    assert hasattr(
        layer_base, "layers"
    ), "The model does not have the expected structure"
    num_layers = len(layer_base.layers)
    print("Applying measurement from layer",layer_target,"of [ 0 ..",(num_layers-1),"]")
    print("to layers",skip_begin_layers,"through",(num_layers - 1) - skip_end_layers)
    for layer_idx in tqdm(
        range(skip_begin_layers, num_layers - skip_end_layers),
        desc="Applying abliteration",
    ):
        layer_base.layers[layer_idx].self_attn.o_proj.weight = modify_tensor_improved2(
            layer_base.layers[layer_idx].self_attn.o_proj.weight.data,
            refusal_dir,
            scaling,
        )
        torch.cuda.empty_cache()
        gc.collect()
        layer_base.layers[layer_idx].mlp.down_proj.weight = modify_tensor_improved2(
            layer_base.layers[layer_idx].mlp.down_proj.weight.data,
            refusal_dir,
            scaling,
        )
        torch.cuda.empty_cache()
        gc.collect()

    return model
