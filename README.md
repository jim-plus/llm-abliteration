# llm-abliteration

Make abliterated models using Transformers, easy and fast. Now faster with batch inference.

## Introduction

There exist directions that cause LLMs to refuse users' input. Abliteration is a technique that can approximate the most significant refusal direction by contrasting harmful and harmless prompts, and then remove/ablate the direction from the model. This is a proof-of-concept implementation to explore the removal refusals from an LLM without the use of TransformerLens, although some GPU acceleration has been implemented.

The code has been tested on Llama-3.2, Qwen2.5-Coder, Ministral-8b, and now Mistral-7B-Instruct-v0.2.

VRAM/RAM requirements: This codebase reflects efforts to reduce VRAM usage. You can abliterate whatever any model provided it fits within VRAM. Loading model in 4-bit precision using bitsandbytes is possible and recommended for large models when VRAM is limited. It is assumed that there is enough cpu memory to load the **bf16** (or full weight) model; the method for ablating the refusal vector could be enhanced to perform lazy-loading in the future to reduce this requirement.

CUDA is assumed to be available, for practical purposes. The original abliteration paper and code used TransformerLens, and measured resid_pre, resid_mid, and resid_post. Failspy's code measured resid_pre and resid_post. Sumandora's code based on Transformers accesses the equivalent of resid_post with hidden_states.

> [!NOTE]
> Abliteration is not full removal of censorship. Abliteration doesn't necessarily mean the model is completely uncensored; a properly abliterated model will not explicitly refuse, theoretically, based on the nature of refusal captured in datasets used for abliteration.

## Quick Start

### Clone the repository

```shell
git clone https://github.com/jim-plus/llm-abliteration.git && cd llm-abliteration
```

### Install dependencies

```shell
pip install -r requirements.txt
```

### Make your abliterations (outdated way)

```shell
python abliterate.py -m <path_to_your_model> -o <output_dir>
```

### Measure harmful, harmless, and refusal directions

```shell
python measure.py -m <path_to_your_model> -o <output_file>
```

### Analyze resulting measurements, with optional charting

```shell
python analyze.py <measurement_file> -c
```

### Abliterate model

```
python sharded_abliteration.py <abliteration_yaml_file>
```

### Chat with your abliterated model

```shell
python chat.py -m <path_to_your_abliterated_model>
```

### Compare between models

```shell
python compare.py -a <model_a> -b <model_b>
```

### Examples

- Abliterate Llama-3.2:

```shell
python abliterate.py -m meta-llama/Llama-3.2-3B-Instruct -o llama3.2-3b-abliterated
```

- Compare your abliterated model with the original model:

```shell
python compare.py -a meta-llama/Llama-3.2-3B-Instruct -b llama3.2-3b-abliterated
```

> [!NOTE]
> The measurement process will autodetect 4-bit and 8-bit BitsAndBytes models. However, abliteration needs to be performed on full-weight models. As a temporary workaround, it's recommended to use `--output-refusal` to dump out the refusal data for later processing.
> The `--input-refusal` argument enables loading of the refusal data from file, skipping measurement; the full-weight model can be specified here. Right now abliteration application loads the entire model into "cpu" memory, a limitation which will need to be fixed before this codebase can handle larger models.

Now your model will be abliterated and saved to `<output_dir>`. Once it finishes, you can immediately chat with your abliterated model in the terminal. For Chinese models, you can use `--deccp` to abliterate it from certain topics.

## Advanced Usage

### Use config files

This repository now supports `.json` config file. This file should contain a `dict` of config key value pairs. For example:

```json
{
    "model": "/absolute/path/to/your/model",
    "output": "/output/dir",
    "data-harmful": "/absolute/path/to/harmful-prompts.txt",
    "scale-factor": 114,
}
```

```shell
python abliterate.py -c config.json
```

Loading config file will **overwrite** command line arguments.

### Use your own prompts

You can use your own prompts to abliterate your model. Supported file formats are `.txt`, `.parquet`, `.json`, and `.jsonl`. Detailed formats are listed below:

- `.txt`: Each line of the file is a prompt
- `.parquet`: A parquet file with column `text`
- `.json`: A JSON file with list of strings
- `.jsonl`: A JSON Lines file with a list of strings

Then load your own prompts using `--data-harmful` and `--data-harmless` arguments:

```shell
python abliterate.py -m <path_to_your_model> -o <output_dir> --data-harmful /path/to/my/harmful.txt --data-harmless /path/to/my/harmless.txt
```

### Scale factor

You can use `--scale-factor` to control the abliteration strength. A scale factor larger then 1 will impose stronger removal of refusals, while a negative scale factor will encourage refusal. You can try to increase the scale factor to see if it helps.

```shell
python abliterate.py -m <path_to_your_model> -o <output_dir> --scale-factor 1.5
```

### Input/Output refusals

You can output the refusals to a file using `--output-refusals` argument:

```shell
python abliterate.py -m <path_to_your_model> -o <output_dir> --output-refusals refusals.bin
```

And load the refusals back using `--load-refusals` argument:

```shell
python abliterate.py -m <path_to_your_model> --input-refusals refusals.bin -o <output_dir>
```

If `--input-refusal` is provided, the script will not compute refusal directions again.

### Abliterate specific targets

By default, abliteration will be applied to `o_proj` and `down_proj`. You can add more targets by modifying the code below, as long as it won't mess up the model:

```python
# utils/apply.py, apply_abliteration()
lm_model.layers[layer_idx].self_attn.o_proj.weight = modify_tensor(
  lm_model.layers[layer_idx].self_attn.o_proj.weight.data,
  refusal_dir,
  scale_factor,
)
lm_model.layers[layer_idx].mlp.down_proj.weight = modify_tensor(
  lm_model.layers[layer_idx].mlp.down_proj.weight.data,
  refusal_dir,
  scale_factor,
)
```

Available targets can be found in [transformers model architectures](https://github.com/huggingface/transformers/tree/main/src/transformers/models) and [mergekit model architectures](https://github.com/arcee-ai/mergekit/tree/main/mergekit/_data/architectures).

### Best practices

This repository provides a bunch of parameters to optimize. To get the best results, you can try the following steps:

1. Carefully choose your prompts. Prompts in this repository are for illustrative purposes only; curate your own prompt datasets to obtain better results.
2. Adjust parameters. The script provides various parameters to control the abliteration process.
3. Experiment with changing the targets. You can modify the code to abliterate other targets, as long as it won't mess up the model.
4. If you have limited VRAM, try load the model as a 4-bit or 8-bit BitsAndBytes quant.

### Full arguments

Use `--help` to see all available arguments:

```shell
python abliterate.py --help
```

## Credits

- [Orion-zhen/abliteration](https://github.com/Orion-zhen/abliteration)
- [Sumandora/remove-refusals-with-transformers](https://github.com/Sumandora/remove-refusals-with-transformers)
- [FailSpy/abliterator](https://github.com/FailSpy/abliterator/)
- [AUGMXNT/deccp](https://github.com/AUGMXNT/deccp)
- [huihui-ai](https://huggingface.co/huihui-ai)
- [Refusal in LLMs is mediated by a single direction](https://github.com/andyrdt/refusal_direction)
