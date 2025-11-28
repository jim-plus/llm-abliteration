import gc
import torch
from tqdm import tqdm
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from utils.device import clear_device_cache

def extract_hidden_states_gpu(raw_output) -> dict:
    processed = {}
    assert hasattr(raw_output, "hidden_states")
    gpu_hidden = []
    for layer_output in raw_output.hidden_states:
        gpu_hidden.append(layer_output)
    processed["hidden_states"] = gpu_hidden
    return processed


def scoring_gpu_batched(
    tokens: list[torch.Tensor],
    desc: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    layer_idx: int,
    refusal_dir: torch.Tensor,
    bias_term: torch.Tensor,
    pos: int = -1,
    batch_size: int = 1,
) -> dict:

    vocab_size = model.config.vocab_size
    refusal_dir = refusal_dir.to(model.device)
    bias_term = bias_term.to(model.device)
    scores = []

    for i in tqdm(range(0, len(tokens), batch_size), desc=desc):
        batch = tokens[i:i+batch_size]
        max_len = max(t.size(1) for t in batch)

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

        batch_input = torch.cat(padded_batch, dim=0).to(model.device)
        batch_mask = torch.cat(attention_masks, dim=0).to(model.device)

        raw_output = model.generate(
#        raw_output = model(
            batch_input,
            attention_mask=batch_mask,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_hidden_states=True,
            pad_token_id=tokenizer.eos_token_id,
        )

        del batch_mask
 #       gpu_output = extract_hidden_states_gpu(raw_output)
        current_hidden = raw_output.hidden_states[0][layer_idx][:, pos, :]
        del raw_output

#        current_hidden = gpu_output["hidden_states"][0][layer_idx][:, pos, :]

        batch_size_actual = current_hidden.size(dim=0)

        for activation in current_hidden:
            # bias term is flipped from abliteration to classification
            # compliance should score positive
            # refusal should score negative
            score = torch.dot(refusal_dir,activation) - bias_term
            scores.append(score.item())

#        del gpu_output, batch_input
        del batch_input
        clear_device_cache()

    return_dict = {}
    return_dict["scores"] = scores
    return return_dict


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

    # 5. Angle between vectors (in degrees)
    angle = torch.acos(torch.clamp(
        torch.tensor(cos_sim), -1.0, 1.0
    )) * 180 / 3.14159

    # 6. Signal quality
    quality = snr * (1 - cos_sim)

    print(f"=== Refusal Direction Analysis (Layer {layer_idx}) ===")
    print(f"Cosine similarity:       {cos_sim:.4f}")
    print(f"Angle between means:     {angle:.2f}°")
    print(f"Harmful mean norm:       {harmful_norm:.4f}")
    print(f"Harmless mean norm:      {harmless_norm:.4f}")
    print(f"Refusal direction norm:  {refusal_norm:.4f}")
    print(f"Signal-to-noise ratio:   {snr:.4f}")
    print(f"Signal quality:          {quality:.4f}")


def score_refusals(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    candidate_list: list[str],
    refusal_dir: torch.Tensor,
    bias_term: torch.Tensor,
    layer_idx: int = -1,
    inference_batch_size: int = 32,
):

    refusal_dir = torch.nn.functional.normalize(refusal_dir, dim=-1)
    print("Tokenizing inputs")
    candidate_tokens = [
        tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": inst}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
        for inst in candidate_list
    ]
    clear_device_cache()
    gc.collect()

    num_layers = len(model.model.layers)
    if layer_idx == -1:
        layer_idx = int(num_layers * 0.6) # default guesstimate
    pos = -1

    results = scoring_gpu_batched(candidate_tokens, "Generating candidate outputs", model, tokenizer, layer_idx, refusal_dir, bias_term, pos, inference_batch_size)

    clear_device_cache()
    gc.collect()

    return results
