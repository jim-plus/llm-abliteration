# llm-abliteration

Make abliterated models using Transformers, easy and fast. Now faster with batch inference.

## Introduction

There exist directions that cause LLMs to refuse users' input. Abliteration is a technique that can approximate the most significant refusal direction by contrasting harmful and harmless prompts, and then remove/ablate the direction from the model. This is a proof-of-concept implementation to explore the removal refusals from an LLM without the use of TransformerLens.

Architectures: Both conventional (dense) and selected MoE architectures are now supported. The code in various forms has been tested on Llama-3.2, Qwen2.5-Coder, Ministral-8b, Mistral-7B-Instruct-v0.2, gemma-3-27b-it, and Mistral-Nemo-Instruct-2407.

VRAM/RAM requirements: This codebase reflects efforts to reduce VRAM usage. You can abliterate whatever any model provided it fits within VRAM. Loading model in 4-bit precision using bitsandbytes is possible and recommended for large models when VRAM is limited.
Ablation loads in one safetensors shard at a time into CPU memory; at most one layer is ablated within VRAM, allowing ablation of large models.

Compute requirements: CUDA or Apple Metal is assumed to be available. CPU-only is impractical due to the inference requirement, though it can theoretically be done.

> [!NOTE]
> Abliteration does not guarantee full removal of censorship, as refusal direction is diffuse. Abliteration doesn't necessarily mean the model is completely uncensored; a properly abliterated model will not explicitly refuse, theoretically, based on the nature of refusals captured in datasets used for abliteration.

The original abliteration paper and code used TransformerLens, measuring resid_pre, resid_mid, and resid_post. Failspy's code measured resid_pre and resid_post. Sumandora's code based on Transformers accesses the equivalent of resid_post with hidden_states; this codebase is ultimately descended from that.

For an explanation of abliteration, see: https://huggingface.co/blog/mlabonne/abliteration

This repo enables norm-preserving biprojected abliteration, a combination of geometric modification to abliteration to reduce the impact on model performance. https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration

Removal of the projected contribution during measurement to orthogonalize the refusal direction is optional, as is removal of the projected contribution during ablation as well as norm preservation. Default functionality is conventional abliteration, and enables independent exploration of the three options. Some models may not respond well to norm presrevation.

## Quick Start

### Clone the repository

```shell
git clone https://github.com/jim-plus/llm-abliteration.git && cd llm-abliteration
```

### Install dependencies

```shell
pip install -r requirements.txt
```

### Workflow

Roughly:
- Measure directions using measure.py, given harmful and harmless prompt datasets
- Analyze directions by layer using analyze.py to determine abliteration strategy
- Craft YAML file to drive ablation
- Ablate model using sharded_ablation.py
- Test resulting model

### Measure harmful, harmless, and refusal directions

```shell
python measure.py -m <path_to_your_model> -o <output_file>
```
Carefully curate your prompt datasets to obtain better results.
You can explicitly specify prompt dataset files, either as local files or on HuggingFace.
```shell
python measure.py -m <path_to_your_model> -o <output_file> --data-harmful DATA_HARMFUL --data-harmless DATA_HARMLESS
```
For Chinese models, you can also specify `--deccp` to add certain topics to the "harmful" set to be evaluated.

The measurement script autodetects 4-bit and 8-bit BitsAndBytes models and will attempt to run on them.
One can also specify `--quant 4bit` and `--quant 8bit` to force on-the-fly bitsandbytes quantization of full-weight models.
However, subsequent ablation needs to be performed on full-weight models.

To orthogonalize the refusal direction against the harmless direction during measurement, specify `--projected`; otherwise the result will correspond to conventional abliteration.

### Analyze resulting measurements, with optional charting

```shell
python analyze.py <measurement_file> -c
```
The `-c` option will put up some nice charting. Look toward middle to late middle layers for good candidate layer sources for refusal direction.

### Abliterate model

```
python sharded_ablate.py <abliteration_yaml_file>
```
Look at the example YAML file to see how this is structured.
YAML was opted for in order to allow more than one source layer for refusal direction measurement, and for different strategies to be applied per destination layer.

To orthogonalize the refusal direction against the harmless direction during measurement, specify `--projected`; otherwise the result will correspond to conventional abliteration.

To preserve weight norms/magnitudes during ablation, specify `--normpreserve`; otherwise the result will correspond to conventional abliteration.

To invert the ablation direction, shifting from ablation to addition, specify `--invert`.

### Chat with your abliterated model

```shell
python chat.py -m <path_to_your_abliterated_model>
```
Inherited code that is in need of an update to remain useful.

### Compare between models

```shell
python compare.py -a <model_a> -b <model_b>
```
Inherited code that is in need of an update to remain useful.

## Advanced Usage

### Use your own prompts

You can use your own prompts to abliterate your model. Supported file formats are `.txt`, `.parquet`, `.json`, and `.jsonl`. Format explanations are below:

- `.txt`: Each line of the file is a prompt
- `.parquet`: A parquet file with column `text`
- `.json`: A JSON file with list of strings
- `.jsonl`: A JSON Lines file with a list of strings

Then load your own prompts using `--data-harmful` and `--data-harmless` arguments during measurement.

Two scripts have been provided to convert between parquet and jsonl formats to assist in dataset customization.
Prompts in this repository are for illustrative purposes only, and have mostly been inherited from the fork.

```shell
python measure.py -m <path_to_your_model> -o <output_file> --data-harmful /path/to/my/harmful.txt --data-harmless /path/to/my/harmless.txt
```
### Tips

If you have limited VRAM, try loading the model as a 4-bit or 8-bit BitsAndBytes quant.

## Credits

- [Orion-zhen/abliteration](https://github.com/Orion-zhen/abliteration)
- [Sumandora/remove-refusals-with-transformers](https://github.com/Sumandora/remove-refusals-with-transformers)
- [FailSpy/abliterator](https://github.com/FailSpy/abliterator/)
- [AUGMXNT/deccp](https://github.com/AUGMXNT/deccp)
- [huihui-ai](https://huggingface.co/huihui-ai)
- [Refusal in LLMs is mediated by a single direction](https://github.com/andyrdt/refusal_direction)

- Thanks to @AesSedai for initial MoE support
- Thanks to @otarkhan for Apple Metal support
