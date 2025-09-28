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


def compute_refusals(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    harmful_list: list[str],
    harmless_list: list[str],
    layer_fraction: float = 0.6,
) -> torch.Tensor:

    def welford(tokens: list[torch.Tensor], desc: str) -> torch.Tensor:
        mean = None
        count = 0
        for token in tqdm(tokens, desc=desc):
            attention_mask = torch.ones_like(token, dtype=torch.long, device=mod
el.device)
            raw_output = model.generate(
                token.to(model.device),
                attention_mask=attention_mask,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_hidden_states=True,
                # use_cache=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            del attention_mask
            cpu_output = extract_hidden_states(raw_output)
            del raw_output
            current_hidden = cpu_output["hidden_states"][0][layer_idx][:, pos, :]
            del cpu_output
            assert isinstance(current_hidden, torch.Tensor)
            # current_hidden.detach()

            batch_size = current_hidden.size(dim=0)
            total_count = count + batch_size

            if mean is None:
                mean = current_hidden.mean(dim=0)
            else:
                delta = current_hidden - mean.squeeze(0)
                mean = mean + (delta.sum(dim=0)) / total_count
            count = total_count

            del current_hidden
            torch.cuda.empty_cache()
        assert mean is not None
        return mean

    harmful_tokens = [
        tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": inst}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
        for inst in harmful_list
    ]
    harmless_tokens = [
        tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": inst}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
        for inst in harmless_list
    ]

    torch.cuda.empty_cache()
    gc.collect()

    layer_idx = int(len(model.model.layers) * layer_fraction) # type: ignore
    pos = -1

    harmful_mean = welford(harmful_tokens, "Generating harmful outputs") # type: ignore
    gc.collect()
    harmless_mean = welford(harmless_tokens, "Generating harmless outputs") # type: ignore
    refusal_dir = harmful_mean - harmless_mean
    refusal_dir = refusal_dir / refusal_dir.norm()
    print(refusal_dir)
    return refusal_dir
