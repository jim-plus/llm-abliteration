from transformers import (
    TextStreamer,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoConfig,
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

    # Resolve precision with sensible defaults
    if args.precision == "auto":
        model_config = AutoConfig.from_pretrained(args.model)
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
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=precision,
            bnb_4bit_use_double_quant=True,
        )
    elif args.load_in_8bit:
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
            llm_int8_has_fp16_weight=True,
        )
    else:
        quant_config = None

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=precision,
        low_cpu_mem_usage=True,
        device_map=device_map,
        quantization_config=quant_config,
        attn_implementation="flash_attention_2" if args.flash_attn and device == "cuda" else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )

    conversation = []
    streamer = TextStreamer(tokenizer)
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
        gen = model.generate(
            **{k: v.to(model.device) for k, v in toks.items()},
            streamer=streamer,
            max_new_tokens=args.max_new_tokens
        )
        decoded = tokenizer.decode(
            gen[0][input_ids.shape[1]:], skip_special_tokens=True
        )
        conversation.append({"role": "assistant", "content": decoded})
