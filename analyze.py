import argparse
import sys
import math
import torch
import matplotlib.pyplot as plt
import heapq
import numpy as np

parser = argparse.ArgumentParser(
    description="Process measurement data and optionally output a charting script.",
    formatter_class=argparse.RawTextHelpFormatter
)

parser.add_argument(
    'data_file',
    metavar='DATA_FILE',
    type=str,
    help='The path to the mandatory input data file containing measurements.'
)

parser.add_argument(
    '-c', '--chart',
    action='store_true',  # Key change: stores True if flag is present, False otherwise
    help="Optional flag to enable charting (generates a default chart file)."
)

args = parser.parse_args()

input_file = args.data_file
should_chart = args.chart

results = {}
config = {}
config['input-refusal'] = input_file
print(f"Loading refusal information from {config['input-refusal']}...")
results = torch.load(config["input-refusal"])
layers = results["layers"]
print("Total layers:",layers)

cosine_similarities = []
cosine_similarities_harmful = []
cosine_similarities_harmless = []
harmful_norms = []
harmless_norms = []
ratio_norms = []
refusal_directions = []
snratios = []
signal_quality_estimates = []
purity_ratios = []

prior_harmful = None
prior_harmless = None
prior_refusal = None

for layer in range(layers):
    harmful_mean = results[f'harmful_{layer}']
    harmless_mean = results[f'harmless_{layer}']
    refusal_dir = results[f'refuse_{layer}']
    precision = refusal_dir.dtype

    # 1. Cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(
        harmful_mean.float(), harmless_mean.float(), dim=0
    ).item()
    cosine_similarities.append(cos_sim)
    cos_sim_harmful = torch.nn.functional.cosine_similarity(
        harmful_mean.float(), refusal_dir.float(), dim=0
    ).item()
    cosine_similarities_harmful.append(cos_sim_harmful)
    cos_sim_harmless = torch.nn.functional.cosine_similarity(
        harmless_mean.float(), refusal_dir.float(), dim=0
    ).item()
    cosine_similarities_harmless.append(cos_sim_harmless)

    # 2. Magnitudes
    harmful_norm = harmful_mean.norm().item()
    harmful_norms.append(harmful_norm)
    harmless_norm = harmless_mean.norm().item()
    harmless_norms.append(harmless_norm)
    ratio_norm = math.log(abs(harmful_norm)/abs(harmless_norm))
    ratio_norms.append(ratio_norm)

    # 3. Refusal direction properties
#    refusal_dir = harmful_mean - harmless_mean
    refusal_norm = refusal_dir.norm().item()
    refusal_directions.append(refusal_norm)

    # 4. Signal-to-noise ratio
    # If means are very similar, refusal_dir is small relative to means
    snr = refusal_norm / max(harmful_norm, harmless_norm)
    snratios.append(snr)

    # 5. Refusal purity
    # Compute raw refusal direction (harmful - harmless)
    raw_refusal_dir = harmful_mean - harmless_mean

    # Normalize the harmless direction for projection
    harmless_normalized = harmless_mean / harmless_mean.norm()

    # Project refusal onto harmless: projection = (refusal · harmless_norm) * harmless_norm
    projection = (refusal_dir @ harmless_normalized) * harmless_normalized

    # Orthogonalized refusal direction (Gram-Schmidt)
    refusal_orth = refusal_dir - projection

    # Compute purity ratio
    if refusal_dir.norm() > 0:
        purity_ratio = refusal_orth.norm() / refusal_dir.norm()
    else:
        purity_ratio = 0
    purity_ratios.append(purity_ratio)

    # 6. Signal quality
    quality = snr * (1 - cos_sim) * purity_ratio
    signal_quality_estimates.append(quality)

    print(f"=== Refusal Direction Analysis (Layer {layer}) ===")
    print(f"Cosine similarity:       {cos_sim:.4f}")
    print(f"Harmful mean norm:       {harmful_norm:.4f}")
    print(f"Harmless mean norm:      {harmless_norm:.4f}")
    print(f"Refusal direction norm:  {refusal_norm:.4f}")
    print(f"Signal-to-noise ratio:   {snr:.4f}")
    print(f"Refusal purity ratio:    {purity_ratio:.4f}")
    print(f"Est. Signal quality:     {quality:.4f}")

    prior_harmful = harmful_mean
    prior_harmless = harmless_mean
    prior_refusal = refusal_dir

signal_quality_derivative = np.gradient(signal_quality_estimates)

top10 = heapq.nlargest(10, enumerate(signal_quality_estimates), key=lambda x: x[1])
top10 = [(i,v.item()) for i,v in top10]
print("Estimated top 10 layers to measure from:")
print(top10)

# After the main loop, compute layer-to-layer refusal direction rotation
refusal_dir_list = [results[f'refuse_{layer}'] for layer in range(len(list(range(layers))))]

rotation_consecutive = []
rotation_from_first = []

for i in range(1, len(refusal_dir_list)):
    cos_consec = torch.nn.functional.cosine_similarity(
        refusal_dir_list[i].float(),
        refusal_dir_list[i-1].float(),
        dim=0
    ).item()
    rotation_consecutive.append(cos_consec)

    cos_from_first = torch.nn.functional.cosine_similarity(
        refusal_dir_list[i].float(),
        refusal_dir_list[0].float(),
        dim=0
    ).item()
    rotation_from_first.append(cos_from_first)

import math
angular_velocity = [math.degrees(math.acos(max(-1.0, min(1.0, c)))) for c in rotation_consecutive]

if (should_chart == False):
    sys.exit(0)

# Create figure with subplots
layers = range(layers)
fig, axes = plt.subplots(3, 2, figsize=(14, 10))
fig.suptitle('Refusal Direction Analysis Across Layers', fontsize=16, fontweight='bold')

# Plot 1: Mean Norms
ax1 = axes[0, 0]
ax1.plot(layers, harmful_norms, 'r-o', label='Harmful Mean', linewidth=2, markersize=4)
ax1.plot(layers, harmless_norms, 'g-s', label='Harmless Mean', linewidth=2, markersize=4)
ax1.plot(layers, refusal_directions, 'b-^', label='Refusal Direction', linewidth=2, markersize=4)
ax1.set_xlabel('Layer', fontsize=11)
ax1.set_ylabel('Norm', fontsize=11)
ax1.set_title('Mean Norms vs Layer', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Cosine Similarity
ax2 = axes[0, 1]
ax2.plot(layers, cosine_similarities, 'purple', label="Harmful to harmless", marker='o', linewidth=2, markersize=4)
ax2.plot(layers, cosine_similarities_harmful, 'red', label="Harmful to refusal", marker='o', linewidth=2, markersize=4)
ax2.plot(layers, cosine_similarities_harmless, 'blue', label="Harmless to refusal", marker='o', linewidth=2, markersize=4)
ax2.set_xlabel('Layer', fontsize=11)
ax2.set_ylabel('Cosine Similarity', fontsize=11)
ax2.set_title('Cosine Similarity vs Layer', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Signal-to-Noise Ratio, Purity Ratio
ax3 = axes[1, 0]
ax3.plot(layers, snratios, 'darkorange', label="Signal to noise", marker='d', linewidth=2, markersize=4)
ax3.plot(layers, purity_ratios, 'darkgreen', label="Refusal purity", marker='d', linewidth=2, markersize=4)
ax3.set_xlabel('Layer', fontsize=11)
ax3.set_ylabel('Ratio', fontsize=11)
ax3.set_title('Signal-to-Noise and Refusal Purity Ratios vs Layer', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Est. Signal Quality
ax4 = axes[1, 1]
ax4.plot(layers, signal_quality_estimates, 'teal', marker='*', label='Signal quality estimate',linewidth=2, markersize=6)
ax4.set_xlabel('Layer', fontsize=11)
ax4.set_ylabel('Est. Signal Quality', fontsize=11)
ax4.set_title('Estimated Signal Quality vs Layer', fontsize=12, fontweight='bold')
ax4.plot(layers, ratio_norms, 'purple', label='Harmful/Harmless Log', linewidth=2, markersize=4)
ax4.plot(layers, signal_quality_derivative, 'orange', label='Signal Quality Gradient', linewidth=2, markersize=4)
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Refusal Direction Rotation
ax5 = axes[2, 0]
layer_range_offset = range(1, len(refusal_dir_list))
ax5.plot(layer_range_offset, rotation_consecutive, 'navy', marker='o', 
         label='Consecutive layer rotation', linewidth=2, markersize=4)
ax5.plot(layer_range_offset, rotation_from_first, 'crimson', marker='o',
         label='Rotation from layer 0', linewidth=2, markersize=4)
ax5.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
ax5.axhline(y=0.0, color='gray', linestyle='--', alpha=0.5)
ax5.set_xlabel('Layer', fontsize=11)
ax5.set_ylabel('Cosine Similarity', fontsize=11)
ax5.set_title('Refusal Direction Rotation Across Layers', fontsize=12, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: Angular Velocity
ax6 = axes[2, 1]
ax6.plot(layer_range_offset, angular_velocity, 'darkorchid', marker='o',
         label='Angular velocity (degrees)', linewidth=2, markersize=4)
ax6.set_xlabel('Layer', fontsize=11)
ax6.set_ylabel('Degrees', fontsize=11)
ax6.set_title('Refusal Direction Angular Velocity', fontsize=12, fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('refusal_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
