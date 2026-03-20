import argparse
import gc
import json
import os
import shutil
import torch
import yaml
from pathlib import Path
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from transformers import AutoConfig
from transformers.utils import cached_file
from utils.device import clear_device_cache, get_preferred_device, synchronize_device


def magnitude_sparsify(tensor: torch.Tensor, fraction: float) -> torch.Tensor:
    """Keep only the top fraction of values by magnitude, zero out the rest."""
    if fraction >= 1.0:
        return tensor
    k = int(tensor.numel() * fraction)
    if k == 0:
        return torch.zeros_like(tensor)

    flat = tensor.flatten()
    threshold = torch.topk(flat.abs(), k, largest=True, sorted=False)[0].min()
    mask = tensor.abs() >= threshold
    return tensor * mask


"""
A warning regarding PyTorch's convention vs. Safetensors storage:

PyTorch nn.Linear layers store weights as [out_features, in_features] - each row is an output neuron's weights
Safetensors (HuggingFace format) stores them as [in_features, out_features] - transposed!
"""


def modify_tensor(
    W: torch.Tensor, intervention_dir: torch.Tensor, scale_factor: float = 1.0,
) -> torch.Tensor:
    """
    Modify weight tensor by ablating intervention direction while preserving row norms.
    Returns a plain tensor (not a Parameter).
    """
    original_dtype = W.dtype
    device = get_preferred_device()

    with torch.no_grad():
        # Move tensors for computation
        W_gpu = W.to(device, dtype=torch.float32, non_blocking=True)
        W_rank = W.dim()
        intervention_dir_gpu = intervention_dir.to(device, dtype=torch.float32, non_blocking=True)

        # Ensure intervention_dir is a 1-dimensional tensor
        if intervention_dir_gpu.dim() > 1:
            intervention_dir_gpu = intervention_dir_gpu.view(-1)

        # Normalize intervention direction
        intervention_normalized = torch.nn.functional.normalize(intervention_dir_gpu, dim=0)

        del intervention_dir_gpu # cleanup

        # Transpose here to convert from safetensors convention
        # Handle Shapes: We want the "Output" dimension to be the last dimension for projection.
        # Intervention Vector lives in the Output Space.

        # Case A: Standard Linear [Out, In] -> Transpose to [In, Out]
        if W_rank == 2:
            W_working = W_gpu.T
        # Case B: Fused Experts [Experts, Out, In] -> Permute to [Experts, In, Out]
        # ex: GPT-OSS-20b
        elif W_rank == 3:
            W_working = W_gpu.permute(0, 2, 1)
        else:
            print(f"Warning: Unsupported tensor shape {W_gpu.shape} - Skipping ablation.")
            return W
        
        del W_gpu   # cleanup

        # Apply ablation
        # Compute dot product of each row with intervention direction
        # [..., Out] @ [Out] -> [...,]
        projection = torch.matmul(W_working, intervention_normalized)

        # Subtract the projection
        # [...,] -> [..., 1] * [Out] -> [..., Out]
        W_working -= scale_factor * (projection.unsqueeze(-1) * intervention_normalized)

        # Transpose here to return safetensors convention
        if W_rank == 2:
            result = W_working.T
        elif W_rank == 3:
            result = W_working.permute(0, 2, 1)

        # Convert back to original dtype and CPU
        result = result.to('cpu', dtype=original_dtype, non_blocking=True)

        # Cleanup
        del intervention_normalized, projection, W_working

        synchronize_device(device)
        clear_device_cache()

    return result.detach().clone()


def modify_tensor_norm_preserved(
    W: torch.Tensor, intervention_dir: torch.Tensor, scale_factor: float = 1.0,
) -> torch.Tensor:
    """
    Modify weight tensor by ablating intervention direction while preserving row norms.
    Returns a plain tensor (not a Parameter).
    """
    original_dtype = W.dtype
    device = get_preferred_device()

    with torch.no_grad():
        # Move tensors for computation
        W_gpu = W.to(device, dtype=torch.float32, non_blocking=True)
        W_rank = W.dim()
        intervention_dir_gpu = intervention_dir.to(device, dtype=torch.float32, non_blocking=True)

        # Ensure intervention_dir is a 1-dimensional tensor
        if intervention_dir_gpu.dim() > 1:
            intervention_dir_gpu = intervention_dir_gpu.view(-1)

        # Normalize intervention direction
        intervention_normalized = torch.nn.functional.normalize(intervention_dir_gpu, dim=0)

        del intervention_dir_gpu # cleanup

        # Transpose here to convert from safetensors convention
        # Handle Shapes: We want the "Output" dimension to be the last dimension for projection.
        # Intervention Vector lives in the Output Space.

        # Case A: Standard Linear [Out, In] -> Transpose to [In, Out]
        if W_rank == 2:
            W_working = W_gpu.T
        # Case B: Fused Experts [Experts, Out, In] -> Permute to [Experts, In, Out]
        # ex: GPT-OSS-20b
        elif W_rank == 3:
            W_working = W_gpu.permute(0, 2, 1)
        else:
            print(f"Warning: Unsupported tensor shape {W_gpu.shape} - Skipping ablation.")
            return W
        
        del W_gpu   # cleanup

        # Decompose weight matrix
        # W_working is [in_features, out_features] or [Experts, in_features, out_features]
        W_norm = torch.norm(W_working, dim=-1, keepdim=True)  # [out_features, 1]
        W_direction = torch.nn.functional.normalize(W_working, dim=-1)  # normalized per output neuron

        del W_working   # cleanup

        # Apply ablation to the DIRECTIONAL component
        # Compute dot product of each row with intervention direction
        projection = torch.matmul(W_direction, intervention_normalized)  # [in_features]

        # Subtract the projection
        W_direction_new = W_direction - scale_factor * (projection.unsqueeze(-1) * intervention_normalized)

        # Re-normalize the adjusted direction
        W_direction_new = torch.nn.functional.normalize(W_direction_new, dim=-1)
        # Double-tap re-normalization — second pass catches residual from near-cancellation
        W_direction_new = W_direction_new - (W_direction_new @ intervention_normalized).unsqueeze(-1) * intervention_normalized
        W_direction_new = torch.nn.functional.normalize(W_direction_new, dim=-1)

        # Recombine: keep original magnitude, use new direction
        W_modified = W_norm * W_direction_new

        # Transpose here to return safetensors convention
        if W_rank == 2:
            result = W_modified.T
        elif W_rank == 3:
            result = W_modified.permute(0, 2, 1)

        # Convert back to original dtype and CPU
        result = result.to('cpu', dtype=original_dtype, non_blocking=True)

        # Cleanup
        del intervention_normalized, projection
        del W_direction, W_direction_new, W_norm, W_modified

        synchronize_device(device)
        clear_device_cache()

    return result.detach().clone()


def modify_tensor_householder(
    W: torch.Tensor,
    src_dir: torch.Tensor,
    tgt_dir: torch.Tensor,
    scale_factor: float = 1.0,
) -> torch.Tensor:
    """
    Modify a weight tensor by either suppressing or rotating the component
    along src_dir, with exact row-norm preservation via a renorm clamp.

    Passing -src_dir for tgt_dir provides geodesic suppression along the
    antipodal arc, respecting great-circle interpolation at all scale values.

    Layout convention:
        Accepts both PyTorch [Out, In] and safetensors [In, Out] layouts
        transparently via internal transpose to [In, Out] (or [Experts, In, Out]
        for MoE tensors) before applying the operation.

    -------------------------------------------------------------------------
    tgt_dir provided  —  Rotation mode (geodesic from src_dir to tgt_dir)
    -------------------------------------------------------------------------
    Applies a proper rotation in the plane spanned by src_dir and tgt_dir
    (Rodrigues' formula), implemented as a pair of rank-1 updates to avoid
    materializing the full [D, D] rotation matrix.

        scale_factor=0.0  →  identity
        scale_factor=1.0  →  full rotation s → t
        intermediate:        great-circle arc between s and t

    Isometric by construction; the renorm clamp is applied as a numerical
    safety net only and is a no-op for well-conditioned inputs.

    Degenerate case (s ≈ t, t_perp_norm < 1e-6): rotation angle is ~0,
    tensor is returned unchanged.

    Args:
        W:            Weight tensor, shape [Out, In] or [In, Out] or
                      [Experts, Out, In].
        src_dir:      Source direction vector, shape [D] or broadcastable.
        tgt_dir:      Target direction vector (rotation mode).
        scale_factor: Interpolation coefficient. Default 1.0.
                          0.0  →  identity
                          1.0  →  full rotation s → t (or suppression at antipode)
                                  for t=-s, nullifies s component
                          2.0  →  reflection through hyperplane orthogonal to s
                      Exact at these three anchor points; geodesic interpolation
                      between them via Rodrigues' formula.

    Returns:
        Modified weight tensor, same shape and dtype as W.
    """
    original_dtype = W.dtype
    device = get_preferred_device()

    with torch.no_grad():
        W_gpu = W.to(device, dtype=torch.float32, non_blocking=True)
        W_rank = W.dim()

        src_dir_gpu = src_dir.to(device, dtype=torch.float32, non_blocking=True)
        if src_dir_gpu.dim() > 1:
            src_dir_gpu = src_dir_gpu.view(-1)
        s = torch.nn.functional.normalize(src_dir_gpu, dim=0)
        del src_dir_gpu

        # Transpose to [In, Out] or [Experts, In, Out] for consistent
        # output-dim projection across both layout conventions.
        if W_rank == 2:
            W_working = W_gpu.T
        elif W_rank == 3:
            W_working = W_gpu.permute(0, 2, 1)
        else:
            print(f"Warning: Unsupported tensor shape {W_gpu.shape} - Skipping.")
            return W
        del W_gpu

        # -----------------------------------------------------------------
        # Rotation path: geodesic from s to t (Rodrigues' formula)
        # Implemented as rank-1 updates — avoids materializing [D, D] R.
        #
        # Trig-free formulation: cos θ and sin θ are derived directly from
        # the dot product and cross-norm of s and t, avoiding the atan2/
        # cos/sin round-trip and its associated floating point error.
        # scale_factor=1.0 (full rotation s → t) is the only supported
        # operating point; other values require angle recovery via trig.
        # -----------------------------------------------------------------
        tgt_dir_gpu = tgt_dir.to(device, dtype=torch.float32, non_blocking=True)
        if tgt_dir_gpu.dim() > 1:
            tgt_dir_gpu = tgt_dir_gpu.view(-1)
        t = torch.nn.functional.normalize(tgt_dir_gpu, dim=0)
        del tgt_dir_gpu

        # cos θ and sin θ derived purely from vector geometry — no trig.
        cos_t = (s @ t).clamp(-1.0, 1.0)
        sin_t = torch.sqrt((1.0 - cos_t ** 2).clamp(min=0.0))

        # cos_t - 1 via half-angle identity: avoids catastrophic cancellation
        # near θ ≈ 0 (s ≈ t) where cos_t ≈ 1.
        cos_t_m1 = -2.0 * ((1.0 - cos_t) / 2.0)

        # Guard antipodal case (t ≈ -s): t_perp collapses to zero.
        is_antipodal = cos_t < -(1.0 - 1e-6)
        t_perp_norm = 0.0
        if not is_antipodal:
            t_perp = t - cos_t * s
            t_perp_norm = t_perp.norm()

        del t

        if not is_antipodal and t_perp_norm < 1e-6:
            # s ≈ t: rotation angle is ~0, nothing to do.
            pass
        else:
            if not is_antipodal:
                e2 = t_perp / t_perp_norm
                # Double-tap: remove residual s component from e2 after
                # float cancellation, then renormalize.
                e2 = e2 - (e2 @ s) * s
                e2 = e2 - (e2 @ s) * s
                e2 = torch.nn.functional.normalize(e2, dim=0)

            # Rodrigues rank-1 update:
            # w_new = w
            #       + (cos_t - 1) * (w·e1)*e1   ← shrink e1 component
            #       + (cos_t - 1) * (w·e2)*e2   ← shrink e2 component
            #       + sin_t       * (w·e1)*e2   ← rotate e1 → e2
            #       - sin_t       * (w·e2)*e1   ← rotate e2 → -e1
            W_norms_sq_before = (W_working * W_working).sum(dim=-1, keepdim=True)
            valid_rows = W_norms_sq_before > 1e-24

            proj_s = W_working @ s

            if is_antipodal:
                # cos_t = -1, sin_t = 0: pure negation of s component.
                # cos_t_m1 = -2, so this is w - 2*(w·s)*s — Householder reflection.
                W_working = W_working + cos_t_m1 * proj_s.unsqueeze(-1) * s
            else:
                proj_e2 = W_working @ e2
                W_working = (
                    W_working
                    + cos_t_m1 * proj_s.unsqueeze(-1)  * s
                    + cos_t_m1 * proj_e2.unsqueeze(-1) * e2
                    + sin_t    * proj_s.unsqueeze(-1)  * e2
                    - sin_t    * proj_e2.unsqueeze(-1) * s
                )

            # Renorm clamp: isometric by construction, so this is a
            # numerical safety net only.
            W_norms_sq_after = (W_working * W_working).sum(dim=-1, keepdim=True)
            renorm = torch.where(
                valid_rows,
                (W_norms_sq_before / W_norms_sq_after).clamp(min=1.0 - 1e-12, max=1.0 + 1e-12).sqrt(),
                torch.ones_like(W_norms_sq_after)
            )
            W_working = W_working * renorm

        # Transpose back to original layout convention.
        if W_rank == 2:
            result = W_working.T
        elif W_rank == 3:
            result = W_working.permute(0, 2, 1)

        result = result.to('cpu', dtype=original_dtype, non_blocking=True)

        del s, W_working
        synchronize_device(device)
        clear_device_cache()

    return result.detach().clone()

def ablate_by_layers_sharded(
    model_name: str,
    measures: dict,
    marching_orders: list,
    output_path: str,
    householder: bool,
    norm_preserve: bool,
    projected: bool,
    ensemble: bool,
    invert: bool,
    large_scale: float,
) -> None:
    """
    Memory-efficient contrastive ablation for sharded models.
    Handles both local paths and HuggingFace Hub models.
    Loads one shard at a time, applies all modifications, then saves.
    """

    # Load config using transformers (handles both local and HF hub)
    print(f"Loading config for {model_name}...")
    config = AutoConfig.from_pretrained(model_name)

    # Determine precision
    if hasattr(config, "torch_dtype"):
        precision = config.torch_dtype
    elif hasattr(config, "dtype"):
        precision = config.dtype
    else:
        precision = torch.float32

    if isinstance(precision, str):
        precision_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        precision = precision_map.get(precision, torch.float32)

    print(f"Model precision: {precision}")

    # Get the safetensors index file (handles cache)
    index_path = cached_file(model_name, "model.safetensors.index.json")
    model_dir = Path(index_path).parent

    print(f"Model directory: {model_dir}")

    with open(index_path) as f:
        index = json.load(f)

    weight_map = index["weight_map"]

    # Find layer prefix
    layer_prefix = None
    for key in weight_map.keys():
        if ".layers." in key and ".self_attn." in key:
            layer_prefix = key.split(".layers.")[0]
            print(f"Detected layer prefix: {layer_prefix}")
            break

    if layer_prefix is None:
        raise ValueError("Could not detect layer structure in model weights")

    # Build a map of which keys in which shards need modification
    shard_modifications = {}  # shard_file -> [(key, layer, measurement, scale, sparsity)]

    for layer, measurement, scale, sparsity in marching_orders:
        layer_base = f"{layer_prefix}.layers.{layer}"
        o_proj_pattern = f"{layer_base}.self_attn.o_proj.weight"

        # 1. Output Projections
        proj_suffixes = ["down_proj", "down_proj.weight"]

        # 2. MLP Blocks
        mlp_indicators = [".mlp.", ".ffn."]

        for key, shard_file in weight_map.items():
            # A. Check for Attention Output (O_Proj)
            if key == o_proj_pattern:
                if shard_file not in shard_modifications:
                    shard_modifications[shard_file] = []
                shard_modifications[shard_file].append((key, layer, measurement, scale, sparsity))
                continue

            # B. Check for MLP Output (Experts, Shared, or Dense)
            if not key.startswith(layer_base):
                continue

            # Must be inside an MLP/MoE block
            if not any(x in key for x in mlp_indicators):
                continue

            # Check suffix (Strict endswith to avoid matching other params like scales/blocks)
            if not any(key.endswith(s) for s in proj_suffixes):
                continue

            # EXCLUDE BIAS (e.g. down_proj_bias or down_proj.bias)
            if "bias" in key:
                continue

            # EXCLUDE QUANTIZATION METADATA (e.g. blocks/scales)
            if "blocks" in key or "scales" in key:
                print(f"Warning: Skipping quantized key {key}. Dequantize model first.")
                continue

            # If we get here, it's a valid target
            if shard_file not in shard_modifications:
                shard_modifications[shard_file] = []
            shard_modifications[shard_file].append((key, layer, measurement, scale, sparsity))

    print(f"\nWill modify {len(shard_modifications)} shards out of {len(set(weight_map.values()))} total")

    os.makedirs(output_path, exist_ok=True)

    # Process each shard
    all_shards = sorted(set(weight_map.values()))

    for shard_file in tqdm(all_shards, desc="Processing shards"):
        shard_path = model_dir / shard_file

        if shard_file in shard_modifications:
            print(f"\nLoading and modifying {shard_file}...")

            # Load the entire shard
            state_dict = load_file(str(shard_path))

            # Apply all modifications for this shard
            for key, layer, measurement, scale, sparsity in shard_modifications[shard_file]:
                if key in state_dict:
                    if "experts" in key:
                        print(f"  Modifying Expert in Layer {layer}: {key}")
                    else:
                        print(f"  Modifying Layer {layer}: {key}")

                    refusal_dir = measures[f'refusenorm_{measurement}'].double()
                    #harmful_dir = measures[f'harmful_{layer}'].double()
                    harmless_dir = measures[f'harmless_{layer}'].double()

                    refusal_dir = torch.nn.functional.normalize(refusal_dir, dim=-1)

                    if ensemble:
                        # apply 3-layer neighborhood ensemble
                        refusal_dir_prior = measures[f'refusenorm_{measurement-1}'].double()
                        refusal_dir_prior = torch.nn.functional.normalize(refusal_dir_prior, dim=-1)
                        refusal_dir_next = measures[f'refusenorm_{measurement+1}'].double()
                        refusal_dir_next = torch.nn.functional.normalize(refusal_dir_next, dim=-1)
                        refusal_dir = torch.nn.functional.normalize(refusal_dir + refusal_dir_prior + refusal_dir_next, dim=-1)
                        del refusal_dir_prior, refusal_dir_next

                    harmless_d = harmless_dir.double()
                    harmless_hat = torch.nn.functional.normalize(harmless_d, dim=-1)

                    if projected:
                        # Here we orthogonalize intervention with respect to harmless direction.
                        # The harmless direction is treated as a boundary condition to maintain.
                        # We compute the second orthogonalized vector from Gram-Schmitt orthonormalization.
                        # Two-pass Gram-Schmidt — second pass catches residual from float cancellation
                        refusal_dir = refusal_dir - (refusal_dir @ harmless_hat) * harmless_hat
                        refusal_dir = refusal_dir - (refusal_dir @ harmless_hat) * harmless_hat

                    # Apply sparsity if requested, then renormalize
                    if sparsity > 0.0:
                        refusal_dir = magnitude_sparsify(refusal_dir, fraction=sparsity)
                        refusal_dir = torch.nn.functional.normalize(refusal_dir + refusal_dir_prior + refusal_dir_next, dim=-1)

                    if invert:
                        scale = -scale
                    
                    # apply global scaling
                    scale *= large_scale

                    # Apply modification
                    if householder:
                        state_dict[key] = modify_tensor_householder(
                            state_dict[key],
                            refusal_dir,
                            -refusal_dir,
                            scale,
                        ).contiguous()                        
                    elif norm_preserve:
                        state_dict[key] = modify_tensor_norm_preserved(
                            state_dict[key],
                            refusal_dir,
                            scale,
                        ).contiguous()
                    else:
                        state_dict[key] = modify_tensor(
                            state_dict[key],
                            refusal_dir,
                            scale,
                        ).contiguous()

                    # Clean up
                    del refusal_dir, harmless_dir
                    gc.collect()

            # Save modified shard
            print(f"  Saving {shard_file}...")
            save_file(state_dict, f"{output_path}/{shard_file}")

            # Clean up
            del state_dict
            gc.collect()
            clear_device_cache()

        else:
            # Just copy unmodified shards (no need to load)
            shutil.copy(str(shard_path), f"{output_path}/{shard_file}")

    # Copy the index file
    print("\nCopying configuration files...")
    shutil.copy(str(index_path), f"{output_path}/model.safetensors.index.json")

    # Copy all config files that exist
    config_files = [
        "config.json", "tokenizer_config.json", "tokenizer.json",
        "special_tokens_map.json", "generation_config.json",
        "tokenizer.model", "vocab.json", "merges.txt", "processor_config.json",
        "added_tokens.json", "preprocessor_config.json", "chat_template.json",
        "chat_template.jinja",
        "configuration_deepseek_v3.py", "modeling_deepseek_v3.py",
        "configuration_glm4_moe.py", "modeling_glm4_moe.py"
    ]

    for file in config_files:
        try:
            src_path = cached_file(model_name, file)
            if src_path and os.path.exists(src_path):
                shutil.copy(src_path, f"{output_path}/{file}")
        except Exception:
            # File doesn't exist, skip it
            pass

    print(f"\nModified model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Memory-efficient sharded ablation script using YAML configuration."
    )

    parser.add_argument(
        'file_path',
        type=str,
        help='Path to a YAML configuration file',
    )
    parser.add_argument(
        '--invert',
        action="store_true",
        default=False,
        help='Invert from ablation to induction',
    )    
    parser.add_argument(
        '--householder',
        action="store_true",
        default=False,
        help='Use Householder reflection to ablate intervention',
    )
    parser.add_argument(
        '--normpreserve',
        action="store_true",
        default=False,
        help='Preserve norms/magnitudes across intervention',
    )
    parser.add_argument(
        '--projected',
        action="store_true",
        default=False,
        help='Orthogonalize, projecting intervention against harmless direction',
    )
    parser.add_argument(
        '--layerensemble',
        action="store_true",
        default=False,
        help='Compute intervention as directional mean from 3 layers centered on intended measurement layer',
    )

    args = parser.parse_args()

    # Load YAML configuration
    with open(args.file_path, 'r') as file:
        ydata = yaml.safe_load(file)

    model_name = ydata.get("model")
    measurement_file = ydata.get("measurements")
    output_dir = ydata.get("output")
    ablations = ydata.get("ablate")
    large_scale = float(ydata.get("scale",1.0))

    print("=" * 60)
    print("SHARDED ABLATION CONFIGURATION")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Measurements: {measurement_file}")
    print(f"Output directory: {output_dir}")
    print(f"Number of ablations: {len(ablations)}")
    print(f"Householder ablation: {args.householder}")
    print(f"Norm preservation: {args.normpreserve}")
    print(f"Projected: {args.projected}")
    print(f"Layer ensemble: {args.layerensemble}")
    print(f"Invert ablation to induction: {args.invert}")
    print("=" * 60)

    # Load measurements
    print(f"\nLoading measurements from {measurement_file}...")
    measures = torch.load(measurement_file)
    print(f"Loaded {len(measures)} measurements")

    # Parse ablation orders, with defaults
    orders = [
        (
            int(item['layer']),
            int(item.get('measurement', item['layer'])),
            float(item.get('scale', 1.0)),
            float(item.get('sparsity', 0.0)),
        )
        for item in ablations
    ]

    print("\nAblation orders:")
    for layer, measurement, scale, sparsity in orders:
        print(f"  Layer {layer}: measurement={measurement}, scale={scale}, sparsity={sparsity}")

    # Perform sharded ablation
    print("\n" + "=" * 60)
    print("STARTING ABLATION")
    print("=" * 60)
    ablate_by_layers_sharded(
        model_name=model_name,
        measures=measures,
        marching_orders=orders,
        output_path=output_dir,
        householder=args.householder,
        norm_preserve=args.normpreserve,
        projected=args.projected,
        ensemble=args.layerensemble,
        invert=args.invert,
        large_scale=large_scale,
    )

    print("\n" + "=" * 60)
    print("ABLATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
