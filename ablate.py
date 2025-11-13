import argparse
import subprocess
import sys
import tempfile
import yaml
import os

def main():
    """
    Main function to orchestrate the measurement and ablation process by calling
    measure.py and sharded_ablate.py as separate processes.
    """
    parser = argparse.ArgumentParser(
        description="Orchestrate model measurement and ablation.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # --- Arguments for both scripts ---
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Local model directory or HuggingFace model ID",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory for the final ablated model",
    )
    parser.add_argument(
        "--ablate-config",
        type=str,
        required=True,
        help="YAML file containing the 'ablate' section with layer configurations.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for measurement (e.g., 'cuda', 'cpu'). Auto-detects if not set.",
    )

    # --- Arguments specific to measure.py ---
    measure_group = parser.add_argument_group('Measurement options (for measure.py)')
    measure_group.add_argument(
        "--quant-measure", "-q",
        type=str,
        choices=["4bit", "8bit"],
        default=None,
        help="Perform measurement using 4-bit or 8-bit quantization."
    )
    measure_group.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for measurement inference."
    )
    measure_group.add_argument(
        "--clip",
        type=float,
        default=1.0,
        help="Fraction of prompt activation to clip by magnitude."
    )
    measure_group.add_argument(
        "--flash-attn",
        action="store_true",
        default=False,
        help="Use Flash Attention 2 during measurement."
    )
    measure_group.add_argument(
        "--data-harmful",
        type=str,
        default=None,
        help="Path to harmful prompts file."
    )
    measure_group.add_argument(
        "--data-harmless",
        type=str,
        default=None,
        help="Path to harmless prompts file."
    )
    measure_group.add_argument(
        "--projected",
        action="store_true",
        default=False,
        help="Use projected refusal directions during measurement."
    )
    measure_group.add_argument(
        "--dtype",
        type=str,
        default="float16",
        help="Data type for measurement computation."
    )

    args = parser.parse_args()

    # --- Step 1: Run Measurement ---
    print("\n" + "="*60)
    print("STEP 1: RUNNING MEASUREMENT")
    print("="*60)

    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".pt") as temp_measure_file:
        measure_output_path = temp_measure_file.name
    
    measure_command = [
        sys.executable, "measure.py",
        "--model", args.model,
        "--output", measure_output_path,
        "--batch-size", str(args.batch_size),
        "--clip", str(args.clip),
        "--dtype", args.dtype,
    ]
    
    # Add optional flags if they are set
    if args.device:
        measure_command.extend(["--device", args.device])
    if args.quant_measure:
        measure_command.extend(["--quant-measure", args.quant_measure])
    if args.flash_attn:
        measure_command.append("--flash-attn")
    if args.data_harmful:
        measure_command.extend(["--data-harmful", args.data_harmful])
    if args.data_harmless:
        measure_command.extend(["--data-harmless", args.data_harmless])
    if args.projected:
        measure_command.append("--projected")

    print(f"Executing command: {' '.join(measure_command)}")
    
    try:
        subprocess.run(measure_command, check=True)
        print("Measurement completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during measurement: {e}", file=sys.stderr)
        os.remove(measure_output_path)
        sys.exit(1)


    # --- Step 2: Prepare and Run Ablation ---
    print("\n" + "="*60)
    print("STEP 2: RUNNING SHARDED ABLATION")
    print("="*60)

    # Load the user-provided ablation config to get the 'ablate' section
    with open(args.ablate_config, 'r') as f:
        ablate_config_data = yaml.safe_load(f)

    # Create the temporary config for sharded_ablate.py
    sharded_config = {
        "model": args.model,
        "measurements": measure_output_path,
        "output": args.output,
        "ablate": ablate_config_data.get("ablate", [])
    }

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".yaml") as temp_sharded_config_file:
        yaml.dump(sharded_config, temp_sharded_config_file)
        sharded_config_path = temp_sharded_config_file.name

    ablate_command = [
        sys.executable, "sharded_ablate.py",
        sharded_config_path
    ]

    print(f"Executing command: {' '.join(ablate_command)}")

    try:
        subprocess.run(ablate_command, check=True)
        print("Ablation completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during ablation: {e}", file=sys.stderr)
    finally:
        # --- Step 3: Cleanup ---
        print("\n" + "="*60)
        print("STEP 3: CLEANUP")
        print("="*60)
        print(f"Removing temporary measurement file: {measure_output_path}")
        os.remove(measure_output_path)
        print(f"Removing temporary config file: {sharded_config_path}")
        os.remove(sharded_config_path)

    print("\n" + "="*60)
    print("ABLATION ORCHESTRATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
