import gc
import torch
from tqdm import tqdm
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

def extract_hidden_states(raw_output) -> dict:
    processed = {}

    assert hasattr(raw_output, "hidden_states")
    cpu_hidden = []
    for layer_output in raw_output.hidden_states:
        layer_tensors = []
        for tensor in layer_output:
            assert isinstance(tensor, torch.Tensor)
            layer_tensors.append(tensor.to("cpu"))
        cpu_hidden.append(layer_tensors)
    processed["hidden_states"] = cpu_hidden

    return processed

def extract_hidden_states_gpu(raw_output) -> dict:
    processed = {}
    assert hasattr(raw_output, "hidden_states")
    gpu_hidden = []
    for layer_output in raw_output.hidden_states:
        gpu_hidden.append(layer_output)
    processed["hidden_states"] = gpu_hidden
    return processed

def welford_gpu_batched(
    tokens: list[torch.Tensor],
    desc: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    layer_idx: int,
    pos: int = -1,
    inference_batch_size: int = 1
) -> torch.Tensor:
    mean = None
    count = 0

    for i in tqdm(range(0, len(tokens), batch_size), desc=desc):
        batch = tokens[i:i+batch_size]

        # Find max length in this batch for padding
        max_len = max(t.size(1) for t in batch)

        # Pad all sequences to max length
        padded_batch = []
        attention_masks = []

        for t in batch:
            pad_len = max_len - t.size(1)
            if pad_len > 0:
                padded = torch.nn.functional.pad(
                    t, (0, pad_len), value=tokenizer.pad_token_id
                )
                mask = torch.cat([
                    torch.ones_like(t),
                    torch.zeros(t.size(0), pad_len, dtype=torch.long)
                ], dim=1)
            else:
                padded = t
                mask = torch.ones_like(t, dtype=torch.long)

            padded_batch.append(padded)
            attention_masks.append(mask)

        # Concatenate into single batch tensor
        batch_input = torch.cat(padded_batch, dim=0).to(model.device)
        batch_mask = torch.cat(attention_masks, dim=0).to(model.device)

        raw_output = model.generate(
            batch_input,
            attention_mask=batch_mask,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_hidden_states=True,
            pad_token_id=tokenizer.eos_token_id,
        )

        del batch_mask
        gpu_output = extract_hidden_states_gpu(raw_output)
        del raw_output
        current_hidden = gpu_output["hidden_states"][0][layer_idx][:, pos, :]
        del gpu_output

        assert isinstance(current_hidden, torch.Tensor)

        batch_size_actual = current_hidden.size(dim=0)
        total_count = count + batch_size_actual

        if mean is None:
            mean = current_hidden.mean(dim=0)
        else:
            delta = current_hidden - mean.squeeze(0)
            mean.add_(delta.sum(dim=0) / total_count)

        count = total_count

        del current_hidden, batch_input
        torch.cuda.empty_cache()

    assert mean is not None
    return mean.to("cpu")

def welford_gpu_batched_multilayer(
    formatted_prompts: list[str],
    desc: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    layer_indices: list[int],  # Changed from single layer_idx
    pos: int = -1,
    batch_size: int = 1
) -> dict[int, torch.Tensor]:
    """Returns dict mapping layer_idx -> mean direction"""

    vocab_size = model.config.vocab_size
#    total_probabilities = torch.zeros(vocab_size).to(model.device)
#    total_prompts_processed = 0

    means = {layer_idx: None for layer_idx in layer_indices}
    counts = {layer_idx: 0 for layer_idx in layer_indices}

    for i in tqdm(range(0, len(formatted_prompts), batch_size), desc=desc):
        batch_prompts = formatted_prompts[i:i+batch_size]

        tokenizer.pad_token = tokenizer.eos_token

        batch_encoding = tokenizer(
            batch_prompts,
            padding=True,
            return_tensors="pt",
        )

        batch_input = batch_encoding['input_ids'].to(model.device)
        batch_mask = batch_encoding['attention_mask'].to(model.device)

        raw_output = model.generate(
            batch_input,
            attention_mask=batch_mask,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_hidden_states=True,
#            output_scores=True,
            pad_token_id=tokenizer.eos_token_id,
        )

#        total_prompts_processed += len(batch_input)
        del batch_input, batch_mask
        hidden_states = raw_output.hidden_states[0]
#        logits = raw_output.scores[0]
        del raw_output
#        probabilities = torch.nn.functional.softmax(logits, dim=-1)
#        del logits
#        total_probabilities += torch.sum(probabilities, dim=0)
#        del probabilities

        # Process all layers at once
        for layer_idx in layer_indices:
            current_hidden = hidden_states[layer_idx][:, pos, :]

            batch_size_actual = current_hidden.size(dim=0)
            total_count = counts[layer_idx] + batch_size_actual

            if means[layer_idx] is None:
                means[layer_idx] = current_hidden.mean(dim=0)
            else:
                delta = current_hidden - means[layer_idx]
                means[layer_idx].add_(delta.sum(dim=0) / total_count)

            counts[layer_idx] = total_count
            del current_hidden

        del hidden_states

        torch.cuda.empty_cache()

    return_dict = {layer_idx: mean.to("cpu") for layer_idx, mean in means.items()}
#    avg_probabilities = total_probabilities / total_prompts_processed
#    return_dict["avg_probabilities"] = avg_probabilities.to("cpu")
    return return_dict

def analyze_refusal_sparsity(harmful_mean, harmless_mean):
    """Understand which dimensions matter for refusal"""

    # Compute difference
    refusal_dir = harmful_mean - harmless_mean

    # Analyze magnitude distribution
    abs_values = torch.abs(refusal_dir)

    # How concentrated is the signal?
    sorted_abs = torch.sort(abs_values, descending=True)[0]

    # Cumulative contribution
    total = sorted_abs.sum()
    cumsum = torch.cumsum(sorted_abs, dim=0)
    cumsum_pct = cumsum / total

    # How many dimensions for 50%, 80%, 95% of the signal?
    dims_50 = (cumsum_pct <= 0.5).sum().item()
    dims_80 = (cumsum_pct <= 0.8).sum().item()
    dims_95 = (cumsum_pct <= 0.95).sum().item()

    print(f"Refusal direction sparsity analysis:")
    print(f"  Top {dims_50} dims (of {len(refusal_dir)}) = 50% of signal")
    print(f"  Top {dims_80} dims = 80% of signal")
    print(f"  Top {dims_95} dims = 95% of signal")
    print(f"  Effective sparsity: {dims_80 / len(refusal_dir) * 100:.1f}%")

    # Gini coefficient (inequality measure)
    sorted_vals = torch.sort(abs_values)[0]
    n = len(sorted_vals)
    index = torch.arange(1, n + 1, dtype=torch.float32)
    gini = (2 * (index * sorted_vals).sum()) / (n * sorted_vals.sum()) - (n + 1) / n
    print(f"  Gini coefficient: {gini:.3f}")
    print(f"    (0 = perfectly uniform, 1 = perfectly concentrated)")

    # If Gini > 0.8: Very sparse signal, top-k might help
    # If Gini < 0.5: Dense signal, sparsifying will hurt

    return {
        'dims_50': dims_50,
        'dims_80': dims_80,
        'dims_95': dims_95,
        'gini': gini.item(),
        'top_dims': torch.topk(abs_values, k=100).indices.tolist()
    }

def analyze_dimension_regions(refusal_dir, chunk_size=512):
    """Check if refusal is concentrated in certain regions of the vector"""
    
    abs_values = torch.abs(refusal_dir)
    num_chunks = len(refusal_dir) // chunk_size
    
    for i in range(num_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        chunk = abs_values[start:end]
        chunk_contribution = chunk.sum() / abs_values.sum()
        
        print(f"Dims {start:4d}-{end:4d}: {chunk_contribution*100:.1f}% of signal")

def analyze_direction(
    harmful_mean,
    harmless_mean,
    layer_idx: int = -1,
):
    # 1. Cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(
        harmful_mean, harmless_mean, dim=0
    ).item()

    # 2. Magnitudes
    harmful_norm = harmful_mean.norm().item()
    harmless_norm = harmless_mean.norm().item()

    # 3. Refusal direction properties
    refusal_dir = harmful_mean - harmless_mean
    refusal_norm = refusal_dir.norm().item()

    # 4. Signal-to-noise ratio
    # If means are very similar, refusal_dir is small relative to means
    snr = refusal_norm / max(harmful_norm, harmless_norm)

    # 5. Signal quality
    quality = snr * (1 - cos_sim)

    print(f"=== Refusal Direction Analysis (Layer {layer_idx}) ===")
    print(f"Cosine similarity:       {cos_sim:.4f}")
    print(f"Harmful mean norm:       {harmful_norm:.4f}")
    print(f"Harmless mean norm:      {harmless_norm:.4f}")
    print(f"Refusal direction norm:  {refusal_norm:.4f}")
    print(f"Signal-to-noise ratio:   {snr:.4f}")
    print(f"Signal quality:          {quality:.4f}")

def format_chats(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    prompt_list: list[str]
):
    result_formatted = [
        tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": inst}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for inst in prompt_list
    ]
    return result_formatted

def compute_refusals(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    harmful_list: list[str],
    harmless_list: list[str],
    layer_idx: int = -1,
    inference_batch_size: int = 32,
    sweep: bool = False,
) -> torch.Tensor:

    print("Formatting inputs")
    harmful_formatted = format_chats(tokenizer=tokenizer, prompt_list=harmful_list)
    harmless_formatted = format_chats(tokenizer=tokenizer, prompt_list=harmless_list)

    num_layers = len(model.model.layers)
    if layer_idx == -1:
        layer_idx = int((num_layers - 1) * 0.6) # default guesstimate
    pos = -1
    print("Focus layer:",layer_idx)
    focus_layers = [layer_idx]
    # option for layer sweep
    if (sweep):
        focus_layers = range(num_layers)

    harmful_means = welford_gpu_batched_multilayer(harmful_formatted, "Generating harmful outputs", model, tokenizer, focus_layers, pos, inference_batch_size)
    torch.cuda.empty_cache()
    gc.collect()
    harmless_means = welford_gpu_batched_multilayer(harmless_formatted, "Generating harmless outputs", model, tokenizer, focus_layers, pos, inference_batch_size)

    results = {}
    # track number of layers in model
    results["layers"] = num_layers

    # compute sweep, because we have all the information we need
    for layer in focus_layers:
        print("Measured layer",layer)
        harmful_mean = harmful_means[layer]
        results[f'harmful_{layer}'] = harmful_mean
        harmless_mean = harmless_means[layer]
        results[f'harmless_{layer}'] = harmless_mean
#        analyze_direction(harmful_mean, harmless_mean, layer)
        refusal_dir = harmful_mean - harmless_mean
        results[f'refuse_{layer}'] = refusal_dir

    # track target layer
    results["layer_idx"] = layer_idx

#    harmful_mean = harmful_means[layer_idx]
#    harmless_mean = harmless_means[layer_idx]
#    refusal_dir = harmful_mean - harmless_mean

    # compute KL divergence of logits in case it offers insight
  #  p_harmful = harmful_means["avg_probabilities"]
  #  p_harmless = harmless_means["avg_probabilities"]
  #  kl_div = torch.nn.functional.kl_div(torch.log(p_harmless), p_harmful, reduction='sum')
  #  print(f"KL divergence: {kl_div.item():.4f}")
    # refusal direction implies a refusal hyperplane
 #   b = -torch.dot(refusal_dir / torch.norm(refusal_dir), (harmful_mean + harmless_mean) / 2)
 #   print(f"Computed normal vector (w) shape: {refusal_dir.shape}")
 #   print(f"Computed bias term (b) value: {b.item()}")

#    results["refusal_dir"] = refusal_dir
 #   results["bias_term"] = b

    torch.cuda.empty_cache()
    gc.collect()
    return results
