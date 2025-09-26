import gc
import sys
import torch
import random
from datasets import load_dataset
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.utils.quantization_config import BitsAndBytesConfig
from utils.data import load_data
from utils.compute import compute_refusals
from utils.apply import apply_abliteration
from utils.arguments import parser, generate_config


if __name__ == "__main__":
    args = parser.parse_args()
    config = generate_config(args)
    assert (
        isinstance(config["model"], str)
        and isinstance(config["skip-begin"], int)
        and isinstance(config["skip-end"], int)
        and isinstance(config["scale-factor"], float)
        and isinstance(config["layer-fraction"], float)
    )
    if config["skip-begin"] < 1:
        raise ValueError("Do not mess with the first layer!")
    if config["layer-fraction"] < 0.0 or config["layer-fraction"] > 1.0:
        raise ValueError("Invalid layer fraction")

    torch.inference_mode()
    torch.set_grad_enabled(False)

    if config["precision"] == "fp16":
        precision = torch.float16
    elif config["precision"] == "bf16":
        precision = torch.bfloat16
    else:
        precision = torch.float32

    if config["load-in-4bit"]:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=precision,
            bnb_4bit_use_double_quant=True,
        )
    elif config["load-in-8bit"]:
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
            llm_int8_has_fp16_weight=True,
        )
    else:
        quant_config = None

    if isinstance(config["data-harmful"], str):
        harmful_list = load_data(config["data-harmful"])
    else:
        harmful_list = load_data("./data/harmful.parquet")
    if isinstance(config["data-harmless"], str):
        harmless_list = load_data(config["data-harmless"])
    else:
        harmless_list = load_data("./data/harmless.parquet")

    if config["deccp"]:
        deccp_list = load_dataset("augmxnt/deccp", split="censored")
        harmful_list += deccp_list["text"] # type: ignore

    if isinstance(config["num-harmful"], int) and config["num-harmful"] > 0:
        harmful_list = random.sample(harmful_list, config["num-harmful"])
    if isinstance(config["num-harmless"], int) and config["num-harmless"] > 0:
        harmless_list = random.sample(harmless_list, config["num-harmless"])

    model = AutoModelForCausalLM.from_pretrained(
        config["model"],
        trust_remote_code=True,
        dtype=precision,
        low_cpu_mem_usage=True,
        device_map=config["device"],
        quantization_config=quant_config,
        attn_implementation="flash_attention_2" if config["flash-attn"] else None,
    )
    model.requires_grad_(False)

    if config["skip-begin"] + config["skip-end"] >= len(model.model.layers):
        raise ValueError("Too many layers to skip.")

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"], trust_remote_code=True, device_map=config["device"]
    )

    if isinstance(config["input-refusal"], str):
        print(f"Loading refusal tensor from {config['input-refusal']}...")
        refusal_dir = torch.load(config["input-refusal"])
    else:
        print("Computing refusal tensor...")
        refusal_dir = compute_refusals(
            model, tokenizer, harmful_list, harmless_list, config["layer-fraction"]
        )

    if isinstance(config["output-refusal"], str):
        print(f"Saving refusal tensor to {config['output-refusal']}...")
        torch.save(refusal_dir, config["output-refusal"])

    if not isinstance(config["output"], str):
        sys.exit(0)

    print("Applying refusal tensor...")

    if config["load-in-4bit"] or config["load-in-8bit"]:
        print("Reloading model with bf16 precision...")
        del model
        torch.cuda.empty_cache()
        gc.collect()
        model = AutoModelForCausalLM.from_pretrained(
            config["model"],
            trust_remote_code=True,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="cpu",
        )

    model = apply_abliteration(
        model,
        refusal_dir,
        config["skip-begin"],
        config["skip-end"],
        config["scale-factor"],
    )
    print(f"Saving abliterated model to {config['output']}...")
    model.save_pretrained(config["output"])
    tokenizer.save_pretrained(config["output"])
