import json
from argparse import Namespace, ArgumentParser

parser = ArgumentParser(description="Make abliterated models")
parser.add_argument(
    "--config", "-c", type=str, default=None, help="Path to config file"
)
parser.add_argument(
    "--model",
    "-m",
    type=str,
    default=None,
    help="Local model directory or huggingface model ID",
)
parser.add_argument(
    "--device",
    "-d",
    type=str,
    choices=["auto", "cuda", "cpu"],
    default="auto",
    help="Target device for processing abliteration; warning: bitsandbytes quantization DOES NOT support CPU",
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=32,
    help="Batch size during inference/calibration; default 32, stick to powers of 2 (higher will use more VRAM)"
)
parser.add_argument("--output", "-o", type=str, default=None, help="Output directory")
parser.add_argument(
    "--skip-begin",
    type=int,
    default=1,
    help="Number of layers to skip at the beginning. Defaults to 1 to avoid messing with the first layer",
)
parser.add_argument(
    "--skip-end", type=int, default=0, help="Number of layers to skip at the end"
)
parser.add_argument(
    "--sparsify",
    type=float,
    default=0.0,
    help="Fraction of refusal path tensor to sparsify by magnitude (think minP)",
)
parser.add_argument(
    "--clip",
    type=float,
    default=1.0,
    help="Fraction of prompt activation to clip by magnitude",
)
parser.add_argument(
    "--layer-fraction",
    type=float,
    default=1.0,
    help="Fraction of layers to use for refusal_dir calculation",
)
parser.add_argument(
    "--scale-factor",
    type=float,
    default=1.0,
    help="Scale factor for ablation; use negative scale-factor to discourage refusal; if abliteration is weak, increasing this may help",
)
parser.add_argument(
    "--flash-attn", action="store_true", default=False, help="Use flash attention 2"
)
parser.add_argument(
    "--data-harmful", "-dhf", type=str, default=None, help="Harmful prompts file"
)
parser.add_argument(
    "--data-harmless", "-dhl", type=str, default=None, help="Harmless prompts file"
)
parser.add_argument(
    "--deccp",
    action="store_true",
    default=False,
    help="For Chinese models, add certain topics",
)
parser.add_argument(
    "--sweep",
    action="store_true",
    default=False,
    help="Perform sweep analysis during calibration",
)

refusals = parser.add_mutually_exclusive_group()
refusals.add_argument(
    "--input-refusal", "-ir", type=str, default=None, help="Input tensor for refusal"
)
refusals.add_argument(
    "--output-refusal", "-or", type=str, default=None, help="Output tensor for refusal"
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


def check_config(config: dict[str, str | int | float | None]) -> None:
    if config["model"] is None:
        raise ValueError("Model name must be specified.")
    if config["output"] is None and config["output-refusal"] is None:
        raise ValueError("Output model or refusal direction must be specified.")
    if config["input-refusal"] is not None and config["output-refusal"] is not None:
        raise ValueError("Do NOT use input and output refusal simultaneously.")
    if config["load-in-4bit"] and config["load-in-8bit"]:
        raise ValueError("Do NOT use 4-bit and 8-bit simultaneously.")


def generate_config(args: Namespace) -> dict[str, str | int | float | None]:
    config = {}

    if args.config is not None:
        with open(args.config, "r") as f:
            config = json.load(f)
        if not isinstance(config, dict):
            raise ValueError("Config file must be a dictionary")

    config.setdefault("model", args.model)
    config.setdefault("device", args.device)
    config.setdefault("batch-size", args.batch_size)
    config.setdefault("output", args.output)
    config.setdefault("skip-begin", args.skip_begin)
    config.setdefault("skip-end", args.skip_end)
    config.setdefault("layer-fraction", args.layer_fraction)
    config.setdefault("sparsify", args.sparsify)
    config.setdefault("clip", args.clip)
    config.setdefault("scale-factor", args.scale_factor)
    config.setdefault("flash-attn", args.flash_attn)
    config.setdefault("data-harmful", args.data_harmful)
    config.setdefault("data-harmless", args.data_harmless)
    config.setdefault("deccp", args.deccp)
    config.setdefault("sweep", args.sweep)
    config.setdefault("input-refusal", args.input_refusal)
    config.setdefault("output-refusal", args.output_refusal)
    config.setdefault("load-in-4bit", args.load_in_4bit)
    config.setdefault("load-in-8bit", args.load_in_8bit)

    check_config(config)

    return config
