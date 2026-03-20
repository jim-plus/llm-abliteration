import gc
import torch
from argparse import ArgumentParser
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import AutoModelForImageTextToText
from transformers import AutoTokenizer
from transformers import AutoProcessor
from transformers import BitsAndBytesConfig
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from utils.data import load_data
from utils.models import has_tied_weights
from utils.clip import magnitude_clip
from utils.device import clear_device_cache, get_preferred_device, resolve_device_map, synchronize_device


def welford_gpu_batched_multilayer_float32(
    formatted_prompts: list[str],
    desc: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    layer_indices: list[int],
    position: int = 1,
    batch_size: int = 1,
    clip: float = 1.0,
    processor = None,  # Add processor parameter
    is_vision_model: bool = False,  # Add flag for vision models
) -> dict[int, torch.Tensor]:
    text_config = model.config
    if hasattr(text_config, "text_config"):
        text_config = text_config.text_config
    vocab_size = text_config.vocab_size

    max_tokens = position

    means = {layer_idx: None for layer_idx in layer_indices}
    counts = {layer_idx: 0 for layer_idx in layer_indices}
    dtype = model.dtype

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    if is_vision_model and processor is not None:
        processor.tokenizer.padding_side = 'left'

    for i in tqdm(range(0, len(formatted_prompts), batch_size), desc=desc):
        batch_prompts = formatted_prompts[i:i+batch_size]

        if is_vision_model and processor is not None:
            # For vision models, use the processor with text-only input
            batch_encoding = processor(
                text=batch_prompts,
                return_tensors="pt",
                padding=True,
            )
        else:
            # For text-only models, use the tokenizer
            batch_encoding = tokenizer(
                batch_prompts,
                padding=True,
                return_tensors="pt",
            )
        
        batch_input = batch_encoding['input_ids'].to(model.device)
        batch_mask = batch_encoding['attention_mask'].to(model.device)

        # Use generate to get hidden states at the first generated token position
        raw_output = model.generate(
            batch_input,
            attention_mask=batch_mask,
            max_new_tokens=max_tokens,
            return_dict_in_generate=True,
            output_hidden_states=True,
            pad_token_id=tokenizer.eos_token_id,
#            do_sample=False,                       # disable sampling
#            top_k=None,
#            top_p=None,
#            cache_implementation=None,
        )
        
        #last_non_pad = batch_mask.sum(dim=1) - 1  # shape: (batch,)
        del batch_input, batch_mask
        #hidden_states = raw_output.hidden_states[max_tokens-1] # Generation step
        hidden_states = [
            layer_tensor.detach().clone() 
            for layer_tensor in raw_output.hidden_states[-1]
        ]
        del raw_output
        #last_non_pad = last_non_pad.to(hidden_states.device)

        # Process layers with Welford in float32
        for layer_idx in layer_indices:
            # Cast to float32 for accumulation
            # Index each sample at its own last non-pad position
#            current_hidden = hidden_states[layer_idx][
#                torch.arange(hidden_states[layer_idx].size(0), device=hidden_states[layer_idx].device),
#                last_non_pad,
#                :
#            ].float() # (batch, hidden)
            # only examine state after last generated position
            current_hidden = hidden_states[layer_idx][:, -1, :].double()
            #current_hidden = hidden_states[layer_idx][:, pos, :].float()
            if (clip < 1.0):
                current_hidden = magnitude_clip(current_hidden, clip)

            batch_size_actual = current_hidden.size(dim=0)
            total_count = counts[layer_idx] + batch_size_actual

            if means[layer_idx] is None:
                # Initialize mean in float64
                means[layer_idx] = current_hidden.double().mean(dim=0)
            else:
                # All operations in float64 (means[layer_idx] is already float64)
                batch_mean = current_hidden.double().mean(dim=0)
                delta = batch_mean - means[layer_idx]
                means[layer_idx] += delta * batch_size_actual / total_count

                #delta = current_hidden.double() - means[layer_idx]
                #means[layer_idx] += delta.sum(dim=0) / total_count

            counts[layer_idx] = total_count
            del current_hidden

        del hidden_states
        #del last_non_pad
        clear_device_cache()

    # Move to CPU
    return_dict = {
        layer_idx: mean.to(device="cpu")
        for layer_idx, mean in means.items()
    }
    del means
    clear_device_cache()
    return return_dict

def format_chats(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    prompt_list: list[str],
    processor = None,
):
    # Use processor's tokenizer if available, otherwise use tokenizer directly
    actual_tokenizer = processor.tokenizer if processor is not None else tokenizer
    
    result_formatted = [
        actual_tokenizer.apply_chat_template(
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
    projected: bool = False,
    inference_batch_size: int = 32,
    clip: float = 1.0,
    processor = None,  # processor parameter
    is_vision_model: bool = False,  # flag for vision models
    token2: bool = False, # measure at second token instead of first
) -> torch.Tensor:
    # dtype = model.dtype
    if hasattr(model, "language_model"):
        layer_base = model.language_model.model
    else:
        layer_base = model.model
        if hasattr(layer_base, "language_model"):
            layer_base = layer_base.language_model
    num_layers = len(layer_base.layers)

    pos = 1
    if token2:
        pos = 2

    focus_layers = range(num_layers) # sweep all layers

    harmful_formatted = format_chats(tokenizer=tokenizer, prompt_list=harmful_list, processor=processor)
    harmful_means = welford_gpu_batched_multilayer_float32(
        harmful_formatted, "Generating harmful outputs", model, tokenizer,
        focus_layers, pos, inference_batch_size, clip, processor, is_vision_model
    )
    clear_device_cache()
    del harmful_formatted
    harmless_formatted = format_chats(tokenizer=tokenizer, prompt_list=harmless_list, processor=processor)
    harmless_means = welford_gpu_batched_multilayer_float32(
        harmless_formatted, "Generating harmless outputs", model, tokenizer, 
        focus_layers, pos, inference_batch_size, clip, processor, is_vision_model
    )
    del harmless_formatted

    results = {}
    results["layers"] = num_layers

    # Keep all results in 32-bit float for analysis/ablation
    for layer in tqdm(focus_layers,desc="Compiling layer measurements"):
        harmful_mean = harmful_means[layer]
        results[f'harmful_{layer}'] = harmful_mean.to(dtype=model.dtype)
        harmless_mean = harmless_means[layer]
        results[f'harmless_{layer}'] = harmless_mean.to(dtype=model.dtype)

        harmful_d = harmful_mean.double()
        harmless_d = harmless_mean.double()

        # Compute raw difference of means in float64 to avoid cancellation at high cosine similarity.
        # Saved unnormalized — normalization is deferred to the ablation phase.
        # Note: once unit-normalized, harmful_hat - harmless_hat is exactly the normal of the
        # Householder reflector that maps harmless_hat onto harmful_hat, giving this direction
        # a clean geometric justification beyond naive contrastive difference of means.

        refusal_dir = harmful_d - harmless_d
        results[f'refuse_{layer}'] = refusal_dir.to(dtype=model.dtype)

        # Householder-inspired alternative of computing difference after normalization
        # Unfortunately, it is inferior numerically even if analytically correct
        #harmful_hat = torch.nn.functional.normalize(harmful_d, dim=0)
        #harmless_hat = torch.nn.functional.normalize(harmless_d, dim=0)
        #refusal_dir = harmful_hat - harmless_hat

        if projected: # semantic meaning: preserve activations along the harmless direction
            # Compute Gram-Schmidt second orthogonal vector/direction to remove harmless direction interference from refusal direction
            # Two-pass Gram-Schmidt — second pass catches residual from float cancellation
            harmless_hat = torch.nn.functional.normalize(harmless_d, dim=0)
            refusal_dir = refusal_dir - (refusal_dir @ harmless_hat) * harmless_hat
            refusal_dir = refusal_dir - (refusal_dir @ harmless_hat) * harmless_hat

        refusal_dir = torch.nn.functional.normalize(refusal_dir, dim=0)

        results[f'refusenorm_{layer}'] = refusal_dir.to(dtype=model.dtype)

    clear_device_cache()
    gc.collect()
    return results


def clean_up() -> None:
    """
    Release VRAM/RAM after measurement is complete.

    Call this after deleting model/tokenizer/results in your code:
        del model, tokenizer, processor, results
        clean_up()

    Note: Callers must delete their own references to objects before calling
    this function. Python's scoping rules mean we cannot delete caller's
    variables from within a function.
    """
    gc.collect()
    synchronize_device()
    clear_device_cache()
    gc.collect()  # Second pass for any refs broken by cache clear
    print("Memory cleared successfully.")


def debug_hook(name):
    def hook(module, input, output):
        if isinstance(output, tuple):
            t = output[0]
        else:
            t = output
        inp = input[0] if isinstance(input, tuple) else input
        inp_max = inp.abs().max().item()
        out_max = t.abs().max().item() if not torch.isnan(t).any() else float('nan')
        print(f"Layer {name}: input_max={inp_max:.4f} | output_max={out_max:.4f}")
    return hook



if __name__ == "__main__":
    parser = ArgumentParser(description="Measure models for analysis and abliteration")
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        required=True,
        help="Local model directory or HuggingFace model ID",
    )
    parser.add_argument(
        "--quant-measure", "-q",
        type=str,
        choices=["4bit", "8bit"],
        default=None,
        help="Perform measurement using 4bit or 8bit bitsandbytes quant"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size during inference/calibration; default 32, stick to powers of 2 (higher will use more VRAM)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        required=True,
        help="Output file for measurements"
    )
    parser.add_argument(
        "--clip",
        type=float,
        default=1.0,
        help="Fraction of prompt activation to clip by magnitude",
    )
    parser.add_argument(
        "--flash-attn",
        action="store_true",
        default=False,
        help="Use Flash Attention 2"
    )
    parser.add_argument(
        "--data-harmful",
        type=str,
        default=None,
        help="Harmful prompts file"
    )
    parser.add_argument(
        "--data-harmless",
        type=str,
        default=None,
        help="Harmless prompts file"
    )
    parser.add_argument(
        "--deccp",
        action="store_true",
        default=False,
        help="For Chinese models, add topics to harmful prompts",
    )
    parser.add_argument(
        "--projected",
        action="store_true",
        default=False,
        help="Remove projection along harmless direction from contrast direction",
    )
    parser.add_argument(
        "--token2",
        action="store_true",
        default=False,
        help="Measure after second token instead of after first token",
    )

    args = parser.parse_args()

    assert (
        isinstance(args.model, str)
        and
        isinstance(args.output, str)
    )

    torch.inference_mode()
    torch.set_grad_enabled(False)

    device = get_preferred_device()
    device_map = resolve_device_map()

    model = args.model
    model_config = AutoConfig.from_pretrained(model)
    model_type = getattr(model_config,"model_type")

    # Get the precision/dtype from config, with proper fallback
    if hasattr(model_config, "torch_dtype") and model_config.torch_dtype is not None:
        precision = model_config.torch_dtype
    elif hasattr(model_config, "dtype") and model_config.dtype is not None:
        precision = model_config.dtype
    else:
        # Fallback to bfloat16 on CUDA (if supported), otherwise float32 on MPS/CPU, float16 on CUDA
        if device == "cuda" and torch.cuda.is_bf16_supported():
            precision = torch.bfloat16
        elif device == "cuda":
            precision = torch.float16
        else:
            precision = torch.float32

    has_vision = False
    if hasattr(model_config,"vision_config"):
        has_vision = True
    model_loader = AutoModelForCausalLM
    if (has_vision):
        model_loader = AutoModelForImageTextToText

    quant_config = None
    qbit = args.quant_measure

    if device == "mps" and qbit:
        print("BitsAndBytes quantization is not supported on MPS; disabling requested quantization.")
        qbit = None

    # autodetect BitsAndBytes quant; overrides option
    if hasattr(model_config,"quantization_config"):
        bnb_config = getattr(model_config, "quantization_config")
        if (bnb_config["load_in_4bit"] == True):
            if device == "mps":
                raise RuntimeError("BitsAndBytes 4-bit models are not supported on MPS. Please use CPU/CUDA or load full-precision weights.")
            qbit = "4bit"
            # Override precision with compute dtype from quant config if available
            if "bnb_4bit_compute_dtype" in bnb_config and bnb_config["bnb_4bit_compute_dtype"]:
                precision = bnb_config["bnb_4bit_compute_dtype"]
        elif (bnb_config["load_in_8bit"] == True):
            if device == "mps":
                raise RuntimeError("BitsAndBytes 8-bit models are not supported on MPS. Please use CPU/CUDA or load full-precision weights.")
            qbit = "8bit"

    # Convert string dtype to torch dtype if needed
    if isinstance(precision, str):
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }
        precision = dtype_map.get(
            precision,
            torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32,
        )

    if qbit == "4bit":
        print(f"Using compute dtype from quant config: {precision}")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=precision,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", # better for QLoRA
        )
    elif qbit == "8bit":
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
#            llm_int8_enable_fp32_cpu_offload=True,
            llm_int8_has_fp16_weight=False,
#            llm_int8_threshold=6.0,
        )    

    if isinstance(args.data_harmful, str):
        harmful_list = load_data(args.data_harmful)
    else:
        harmful_list = load_data("./data/harmful.parquet")
    if isinstance(args.data_harmless, str):
        harmless_list = load_data(args.data_harmless)
    else:
        harmless_list = load_data("./data/harmless.parquet")

    if args.deccp:
        deccp_list = load_dataset("augmxnt/deccp", split="censored")
        harmful_list += deccp_list["text"]

    attn_impl = "flash_attention_2" if args.flash_attn and device == "cuda" else None

    if hasattr(model_config, "quantization_config"):
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
#            trust_remote_code=True,
            dtype=precision,
            device_map=device_map,
            attn_implementation=attn_impl,
        )
    else:
        model = model_loader.from_pretrained(
            args.model,
#            trust_remote_code=True,
            dtype=precision,
            low_cpu_mem_usage=True,
            device_map=device_map,
            quantization_config=quant_config,
            attn_implementation=attn_impl,
        )
    model.requires_grad_(False)
    if has_tied_weights(model_type):
        model.tie_weights()

    # point to base of language model
    if hasattr(model, "language_model"):
        layer_base = model.language_model.model
    else:
        layer_base = model.model
        if hasattr(layer_base, "language_model"):
            layer_base = layer_base.language_model


#    for i, layer in enumerate(layer_base.layers):
#        layer.register_forward_hook(debug_hook(i))

    #print(layer_base.embed_tokens.weight.dtype)
    #print(layer_base.config.hidden_size) 

    if qbit == "4bit": # stabilize for Gemma 3, possibly other models
        layer_base.embed_tokens = layer_base.embed_tokens.to(precision)
        layer_base.norm = layer_base.norm.to(precision)
    # Gemma 3 still needs Winsorization to not explode!

    # Load processor for vision models, tokenizer for text-only models
    processor = None
    if has_vision:
        try:
            processor = AutoProcessor.from_pretrained(
                args.model,
                device_map=device_map,
                padding=True,
            )
            tokenizer = processor.tokenizer
            print("Loaded processor for vision model")
        except (IndexError, Exception) as e:
            # If processor loading fails, fall back to tokenizer only
            print(f"Could not load processor ({e}), falling back to tokenizer only")
            has_vision = False
            tokenizer = AutoTokenizer.from_pretrained(
                args.model,
#                trust_remote_code=True,
                device_map=device_map,
                padding=True,
            )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
#            trust_remote_code=True,
            device_map=device_map,
            padding=True,
        )

    print("Computing refusal information...")
    results = {}
    results = compute_refusals(
        model=model,
        tokenizer=tokenizer,
        harmful_list=harmful_list,
        harmless_list=harmless_list,
        projected=args.projected,
        inference_batch_size=args.batch_size,
        clip=args.clip,
        processor=processor,
        is_vision_model=has_vision,
        token2=args.token2,
    )

    print(f"Saving refusal information to {args.output}...")
    torch.save(results, args.output)

    # Release VRAM so next measurement can start immediately
    print("Unloading model and clearing memory...")
    del model, tokenizer, processor, results
    clean_up()
