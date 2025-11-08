import gc
import sys
import torch
import random
from datasets import load_dataset
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import AutoModelForImageTextToText
from transformers import AutoProcessor
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig
from utils.data import load_data
from utils.compute import compute_refusals
from utils.apply import apply_abliteration
from utils.arguments import parser, generate_config
#from utils.sparsify import magnitude_sparsify, sparsity_stats
from utils.models import has_tied_weights


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

    model_config = AutoConfig.from_pretrained(config["model"])
    family = getattr(model_config,"model_type")

    if hasattr(model_config,"dtype"):
        native_precision = getattr(model_config,"dtype")
    elif hasattr(model_config,"torch_dtype"):
        native_precision = getattr(model_config,"torch_dtype")

    has_vision = False
    if hasattr(model_config,"vision_config"):
        has_vision = True
    model_loader = AutoModelForCausalLM
    if (has_vision):
        model_loader = AutoModelForImageTextToText

    precision = getattr(model_config, "dtype")

    quant_config = None
    # autodetect BitsAndBytes quant
    if hasattr(model_config,"quantization_config"):
        bnb_config = getattr(model_config, "quantization_config")
        if (bnb_config["load_in_4bit"] == True):
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=precision,
                bnb_4bit_use_double_quant=True,
            )
        elif (bnb_config["load_in_8bit"] == True):
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
                llm_int8_has_fp16_weight=True,
            )

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
        harmful_list += deccp_list["text"]

    if hasattr(model_config, "quantization_config"):
        model = AutoModelForCausalLM.from_pretrained(
            config["model"],
#            trust_remote_code=True,
            dtype=precision,
            device_map=config["device"],
            attn_implementation="flash_attention_2" if config["flash-attn"] else None,
        )
    else:
        model = model_loader.from_pretrained(
            config["model"],
#            trust_remote_code=True,
            dtype=precision,
            low_cpu_mem_usage=True,
            device_map=config["device"],
            quantization_config=quant_config,
            attn_implementation="flash_attention_2" if config["flash-attn"] else None,
        )
    model.requires_grad_(False)
    if has_tied_weights(family):
        model.tie_weights()

    layer_base = model.model
    if hasattr(layer_base,"language_model"):
        layer_base = layer_base.language_model
    num_layers = len(layer_base.layers)

    if config["skip-begin"] + config["skip-end"] >= num_layers:
        raise ValueError("Too many layers to skip.")

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"],
#        trust_remote_code=True,
        device_map=config["device"],
        padding=True,
    )
# Mistral Small 3.2 24B may be missing tokenizer.chat_template

    layer_idx = int((num_layers - 1) * config["layer-fraction"])

    results = {}
    if isinstance(config["input-refusal"], str):
        print(f"Loading refusal information from {config['input-refusal']}...")
        results = torch.load(config["input-refusal"])
    else:
        print("Computing refusal information...")
        results = compute_refusals(
            model, tokenizer, harmful_list, harmless_list, layer_idx,
            config["batch-size"], config["sweep"], config["clip"]
        )
    #refusal_dir = results["refusal_dir"]
    refusal_dir = results[f'refuse_{layer_idx}']
    #bias_term = results["bias_term"]

    if isinstance(config["output-refusal"], str):
        print(f"Saving refusal information to {config['output-refusal']}...")
        torch.save(results, config["output-refusal"])

#    if not isinstance(config["output"], str):
    if config["output"] is None:
        sys.exit(0)

    print("Ablating refusal direction...")

    model_config = AutoConfig.from_pretrained(config["model"])
    if hasattr(model_config, "quantization_config"):
        print("Ablation of quantized models is unsupported.")
        sys.exit(0)

    if config["load-in-4bit"] or config["load-in-8bit"] or isinstance(config["input-refusal"], str):
        precision = getattr(model_config, "dtype")
        print("Reloading model with",precision,"precision...")
        del model
        torch.cuda.empty_cache()
        gc.collect()
        model = model_loader.from_pretrained(
            config["model"],
#            trust_remote_code=True,
            dtype=precision,
            low_cpu_mem_usage=True,
            device_map="cpu",
        )
        if has_tied_weights(family):
            model.tie_weights()

    model = apply_abliteration(
        model,
        refusal_dir,
        layer_idx,
        config["skip-begin"],
        config["skip-end"],
        config["scale-factor"],
        config["sparsify"],
    )
    print(f"Saving abliterated model to {config['output']}...")
    model.save_pretrained(config["output"])
    tokenizer.save_pretrained(config["output"])
    if (has_vision):
        processor = AutoProcessor.from_pretrained(
            config["model"],
            device_map="cpu"
        )
        print("Saving processor to",config["output"])
        processor.save_pretrained(config["output"])
