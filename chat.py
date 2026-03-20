from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    AutoProcessor,
    BitsAndBytesConfig,
    GenerationConfig,
    TextStreamer,
)
from argparse import ArgumentParser
import torch
from utils.device import get_preferred_device, resolve_device_map

parser = ArgumentParser()
parser.add_argument(
    "--model", "-m", type=str, required=True, help="Path to model directory"
)
parser.add_argument(
    "--precision",
    "-p",
    type=str,
    default="auto",
    choices=["auto", "fp16", "bf16", "fp32"],
    help="Precision to load the model. 'auto' tries model config, then chooses a safe default for the device.",
)
parser.add_argument(
    "--device",
    "-d",
    type=str,
    choices=["auto", "cuda", "cpu", "mps"],
    default="auto",
    help="Target device to process abliteration. Warning, bitsandbytes quantization DOES NOT support CPU",
)
parser.add_argument(
    "--max-new-tokens", "-n", type=int, default=256, help="Max new tokens to generate"
)
quant = parser.add_mutually_exclusive_group()
quant.add_argument(
    "--load-in-4bit",
    action="store_true",
    default=False,
    help="Load model in 4-bit precision using bitsandbytes",
)
quant.add_argument(
    "--load-in-8bit",
    action="store_true",
    default=False,
    help="Load model in 8-bit precision using bitsandbytes",
)
parser.add_argument(
    "--flash-attn", action="store_true", default=False, help="Use flash attention 2"
)
args = parser.parse_args()


if __name__ == "__main__":
    device = get_preferred_device(args.device)
    device_map = resolve_device_map(args.device)
    
    torch.inference_mode()
    torch.set_grad_enabled(False)

    model_config = AutoConfig.from_pretrained(args.model)

    # Resolve precision with sensible defaults
    if args.precision == "auto":
        precision = getattr(model_config, "torch_dtype", None) or getattr(
            model_config, "dtype", None
        )
        if precision is None: # use current popular defaults
            if device == "cuda":
                if torch.cuda.is_bf16_supported():
                    precision = torch.bfloat16
                else:
                    precision = torch.float16
            elif device == "mps":
                precision = torch.float32  # float16 on MPS is unstable
            else:
                precision = torch.float32
    elif args.precision == "fp16":
        precision = torch.float16
    elif args.precision == "bf16":
        precision = torch.bfloat16
    elif args.precision == "fp32":
        precision = torch.float32

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

    # Avoid fp16 instability on MPS unless user forced fp16 explicitly
    if device == "mps" and precision == torch.float16:
        print("! Switching MPS to float32 for stability (fp16 can produce NaNs on MPS).")
        precision = torch.float32

    # MPS cannot run bitsandbytes; keep old CUDA/CPU behavior untouched
    if device == "mps" and (args.load_in_4bit or args.load_in_8bit):
        raise RuntimeError("BitsAndBytes quantization is not supported on MPS. Please disable --load-in-4bit/--load-in-8bit or use CUDA.")

    if args.load_in_4bit:
        print("Loading in 4-bit...")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=precision,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="fp4", # fp4 should be better for activations than np4
#            llm_int8_skip_modules=None,
        )
    elif args.load_in_8bit:
        print("Loading in 8-bit...")
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
#            llm_int8_enable_fp32_cpu_offload=True,
            llm_int8_has_fp16_weight=False,
        )
    else:
        quant_config = None

    attn_impl = None
    if args.flash_attn and device == "cuda":
        attn_impl = "flash_attention_2"

    has_vision = False
    if hasattr(model_config,"vision_config"):
        has_vision = True
    model_loader = AutoModelForCausalLM
    if (has_vision):
        model_loader = AutoModelForImageTextToText

    if hasattr(model_config, "quantization_config") and quant_config is not None:
        print("Warning: model already has a quantization_config; --load-in-4bit/8bit flag may be ignored")

    print("precision",precision)
    print("device map",device_map)
    print("quant config",quant_config)
    model = model_loader.from_pretrained(
        args.model,
        torch_dtype=precision,
        low_cpu_mem_usage=True,
        device_map=device_map,
        quantization_config=quant_config,
        attn_implementation=attn_impl,
    )

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
            # If processor loading fails, fall back to tokenizer
            print(f"Could not load processor ({e}), falling back to tokenizer")
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

    # sampler settings can be placed here
    gen_config = GenerationConfig(
        do_sample=False,
        max_new_tokens=args.max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )

    conversation = []
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    print("Type /clear to clear history, /exit to quit.")
    while True:
        prompt = input("User> ")
        if prompt == "/clear":
            conversation = []
            print("! History cleared.")
            continue
        elif prompt == "/exit":
            break
        elif prompt == "":
            print("! Please type a message.")
            continue
        conversation.append({"role": "user", "content": prompt})
        toks = tokenizer.apply_chat_template(
            conversation=conversation,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        input_ids = toks["input_ids"]
        with torch.inference_mode():
            gen = model.generate(
                **{k: v.to(model.device) for k, v in toks.items()},
                streamer=streamer,
                generation_config=gen_config,
            )
        decoded = tokenizer.decode(
            gen[0][input_ids.shape[1]:], skip_special_tokens=True
        )
        conversation.append({"role": "assistant", "content": decoded})
