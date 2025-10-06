import torch
import argparse
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def quantize_model(model_id, output_dir, quantization_bits):
    """
    Quantize a Hugging Face model to 4-bit or 8-bit and save it.
    
    Args:
        model_id: Hugging Face model identifier (e.g., "mistralai/Mistral-7B-Instruct-v0.2")
        output_dir: Directory where the quantized model will be saved
        quantization_bits: Either "4bit" or "8bit"
    """
    
    # Create the appropriate quantization configuration
    if quantization_bits == "4bit":
        print("Configuring 4-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    else:  # 8bit
        print("Configuring 8-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
    
    # Load the model with quantization
    print(f"Loading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # Load the tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    print(f"Model has been loaded and quantized to {quantization_bits}.")
    print(f"Memory footprint: {model.get_memory_footprint() / 1e9:.2f} GB")
    
    # Save the model and tokenizer
    print(f"Saving quantized model to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"✓ Quantized model and tokenizer saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Quantize a Hugging Face model to 4-bit or 8-bit precision.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python quantize_model.py mistralai/Mistral-7B-Instruct-v0.2 ./output-4bit 4bit
  python quantize_model.py grimjim/mistralai-Mistral-Nemo-Instruct-2407 ./output-8bit 8bit
        """
    )
    
    parser.add_argument(
        "model_name",
        type=str,
        help="Hugging Face model identifier (e.g., 'mistralai/Mistral-7B-Instruct-v0.2')"
    )
    
    parser.add_argument(
        "output_dir",
        type=str,
        help="Output directory where the quantized model will be saved"
    )
    
    parser.add_argument(
        "quantization",
        type=str,
        choices=["4bit", "8bit"],
        help="Quantization precision: either '4bit' or '8bit'"
    )
    
    args = parser.parse_args()
    
    try:
        quantize_model(args.model_name, args.output_dir, args.quantization)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
