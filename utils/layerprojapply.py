import gc
import torch
from tqdm import tqdm
from transformers import PreTrainedModel
from utils.sparsify import magnitude_sparsify

def modify_tensor(
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

def modify_tensor_norm_preserved(
    W: torch.Tensor, refusal_dir: torch.Tensor, scale_factor: float = 1.0,
) -> torch.nn.Parameter:
    original_device = W.device
    original_dtype = W.dtype

    # Use no_grad context to prevent graph construction
    with torch.no_grad():
        # Move tensors to GPU for computation
        W_gpu = W.to('cuda', dtype=torch.float32, non_blocking=True)
        refusal_dir_gpu = refusal_dir.to('cuda', dtype=torch.float32, non_blocking=True)

        # Normalize refusal direction
        refusal_normalized = torch.nn.functional.normalize(refusal_dir_gpu, dim=0)

        # Decompose weight matrix
        W_norm = torch.norm(W_gpu, dim=1, keepdim=True)  # [out_features, 1]
        W_direction = torch.nn.functional.normalize(W_gpu, dim=1)  # normalized per output neuron
    
        # Apply abliteration to the DIRECTIONAL component
        projection = torch.matmul(refusal_normalized, W_direction)  # [in_features]
        W_direction_new = W_direction - scale_factor * torch.outer(refusal_normalized, projection)
    
        # Re-normalize the adjusted direction to enable recombination
        W_direction_new = torch.nn.functional.normalize(W_direction_new, dim=1)
    
        # Recombine: keep original magnitude, use new direction
        W_gpu = W_norm * W_direction_new
        
        # Convert back to original dtype and move back to original device
        tensor_modified = W_gpu.to(original_device, dtype=original_dtype, non_blocking=True)

        del projection
        del refusal_dir_gpu
        del refusal_normalized
        del W_direction
        del W_direction_new
        del W_gpu
        del W_norm
        
        # Force synchronization and clear cache
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # Create a fresh tensor with no computation history
    # .detach() ensures no gradient tracking, .clone() breaks all references
    clean_tensor = tensor_modified.detach().clone()
    del tensor_modified
    
    # Create Parameter from the clean tensor
    return torch.nn.Parameter(clean_tensor)  # Defaults to requires_grad=True

def ablate_by_layers(
    model: PreTrainedModel,
    measures: dict,
    marching_orders: list
) -> PreTrainedModel:
    layer_base = model.model
    precision = model.dtype
    if hasattr(layer_base,"language_model"):
        layer_base = layer_base.language_model
    num_layers = len(layer_base.layers)
    for layer, measurement, scale, sparsity in marching_orders:
        print(layer, measurement, scale, sparsity)
        print(f"Applying measurement from layer {measurement} of [0..{num_layers-1}] to layer {layer}")
        refusal_dir = measures[f'refuse_{measurement}']
        harmless_dir = measures[f'harmless_{layer}']
        # Normalize harmless_mean to avoid numerical issues
        harmless_normalized = torch.nn.functional.normalize(harmless_dir.float(), dim=0)
    
        # Project and subtract
        projection_scalar = refusal_dir.float() @ harmless_normalized
        refined_refusal_dir = refusal_dir.float() - projection_scalar * harmless_normalized
        refusal_dir = refined_refusal_dir.to(precision)
        if sparsity > 0.0:
            refusal_dir = magnitude_sparsify(refusal_dir, fraction=sparsity)
        refusal_dir = torch.nn.functional.normalize(refusal_dir, dim=-1)
        layer_base.layers[layer].self_attn.o_proj.weight = modify_tensor_norm_preserved(
            layer_base.layers[layer].self_attn.o_proj.weight.data,
            refusal_dir,
            scale,
        )
        gc.collect()
        layer_base.layers[layer].mlp.down_proj.weight = modify_tensor_norm_preserved(
            layer_base.layers[layer].mlp.down_proj.weight.data,
            refusal_dir,
            scale,
        )
        del refusal_dir
        gc.collect()

    return model
