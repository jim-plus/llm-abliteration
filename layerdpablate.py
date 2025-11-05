import argparse
import torch
import yaml
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import AutoModelForImageTextToText
from transformers import AutoProcessor
from transformers import AutoTokenizer
from utils.layerprojapply import ablate_by_layers
from utils.models import has_tied_weights

parser = argparse.ArgumentParser(
    description="A fine-grained ablation script that takes a mandatory YAML file path as an argument."
)

parser.add_argument(
    'file_path',
    type=str,
    help='a path to a YAML file',
)

args = parser.parse_args()

yfile = args.file_path

with open(yfile,'r') as file:
    ydata = yaml.safe_load(file)

model_name = ydata.get("model")
print("Model:",model_name)

print("Loading model configuration")
model_config = AutoConfig.from_pretrained(model_name)
family = getattr(model_config,"model_type")
if hasattr(model_config,"dtype"):
    precision = getattr(model_config,"dtype")
elif hasattr(model_config,"torch_dtype"):
    precision = getattr(model_config,"torch_dtype")
print("Loading model with",precision,"precision")
has_vision = False
processor = None
if hasattr(model_config,"vision_config"):
    has_vision = True
model_loader = AutoModelForCausalLM
if (has_vision):
    model_loader = AutoModelForImageTextToText
model = model_loader.from_pretrained(
    model_name,
#    trust_remote_code=True,
    dtype=precision,
    low_cpu_mem_usage=True,
    device_map="cpu",
)
model.requires_grad_(False)
if has_tied_weights(family):
    model.tie_weights()

print("Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    device_map="cpu"
)
if has_vision:
    print("Loading processor")
    processor = AutoProcessor.from_pretrained(
        model_name,
        device_map="cpu"
    )

measurement_file = ydata.get("measurements")

print("Loading measurements from",measurement_file)
measures = torch.load(measurement_file)

output_dir = ydata.get("output")
print("Output directory:", output_dir)

ablations = ydata.get("ablate")
print(ablations)

orders = [
    (
        int(item['layer']),
        int(item['measurement']),
        float(item['scale']),
        float(item['sparsity']),
    )
    for item in ablations
]

print("Applying layerwise interventions")
model = ablate_by_layers(model,measures,orders)

print("Saving abliterated model and tokenizer to",output_dir)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
if has_vision:
    print("Saving processor to",output_dir)
    processor.save_pretrained(output_dir)
