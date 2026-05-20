#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------
# Usage:
#   bash run_dataset.sh <INPUT_DIR> <GT_DIR> <OUTPUT_DIR> <SOURCE_PROMPTS_TXT> <TARGET_PROMPTS_TXT>
# Or just edit the defaults below and run:
#   bash run_dataset.sh
# ---------------------------------------------------------

# Positional args (with sane fallbacks)
INPUT_DIR="${1:-/scratch/gilbreth/lwickrem/data/LtF_test_degrads/superres_8x}"   # degraded images (e.g., .../denoise/)
GT_DIR="${2:-/scratch/gilbreth/lwickrem/data/LtF_test_gt}"             # ground-truth images (RGB), same order
OUTPUT_DIR="${3:-/scratch/gilbreth/lwickrem/RF_Inversion/RF-Solver-Edit/FLUX_Image_Edit/results/ECCV/superres_8x_ddnm_LtF}"  # where to save outputs
SOURCE_PROMPTS_TXT="${4:-/scratch/gilbreth/lwickrem/data/LtF_test_degrads/superres_8x/source_prompts.txt}"
TARGET_PROMPTS_TXT="${5:-/scratch/gilbreth/lwickrem/data/LtF_test_degrads/superres_8x/target_prompts.txt}"
# SOURCE_IMG_PATH="${6:-/scratch/gilbreth/lwickrem/data/afhq_degrads/color/cat_selected_1/flickr_cat_000008.jpg}"

# Script + common args
PYTHON_SCRIPT="edit_dataset.py"
NAME="flux-dev"
NUM_STEPS=30
COMMON_ARGS=(--num_steps "$NUM_STEPS" --name "$NAME")

# --------------------------------
# Hyperparameters (dataset-wide)
# --------------------------------
GUIDANCE=4
INJECT=9
DDNM_INJECT=1
DEGRADATION="super resolution"     # e.g., "colorization" | "super resolution" | "deblurring"| "denoising" | "old photo restoration" | "inpainting"

# --------------------------------
# Batch options
# --------------------------------
RECURSIVE=false                # true to recurse through subfolders
EXTS="jpg,jpeg,png"            # extensions to include
OFFLOAD=true                   # true -> pass --offload
ID_METRIC="clip"               # 'clip' | 'arcface' | 'none'
METRICS_RESIZE="gt_to_pred"    # 'gt_to_pred' | 'pred_to_gt' | 'none'
METRICS_CSV="$OUTPUT_DIR/metrics.csv"

mkdir -p "$OUTPUT_DIR"

# Build extra flags from booleans
EXTRA_ARGS=()
[[ "$OFFLOAD" == "true" ]]   && EXTRA_ARGS+=(--offload)
[[ "$RECURSIVE" == "true" ]] && EXTRA_ARGS+=(--recursive)


echo "=== RF-Edit batch run ==="
echo "Input dir        : $INPUT_DIR"
echo "GT dir           : $GT_DIR"
echo "Source prompts   : $SOURCE_PROMPTS_TXT"
echo "Target prompts   : $TARGET_PROMPTS_TXT"
echo "Output dir       : $OUTPUT_DIR"
echo "Degradation      : $DEGRADATION"
echo "Guidance/Inject  : $GUIDANCE / $INJECT (ddnm_inject=$DDNM_INJECT)"
echo "Offload          : $OFFLOAD"
echo "Exts             : $EXTS"
echo "Metrics CSV      : $METRICS_CSV"
echo "Metrics resize   : $METRICS_RESIZE"
echo

python "$PYTHON_SCRIPT" \
  --input_dir "$INPUT_DIR" \
  --gt_dir "$GT_DIR" \
  --source_prompts_file "$SOURCE_PROMPTS_TXT" \
  --target_prompts_file "$TARGET_PROMPTS_TXT" \
  --output_dir "$OUTPUT_DIR" \
  --guidance "$GUIDANCE" \
  --inject "$INJECT" \
  --ddnm_inject "$DDNM_INJECT" \
  --degradation "$DEGRADATION" \
  "${COMMON_ARGS[@]}" \
  --compute_metrics \
  --id_metric "$ID_METRIC" \
  --metrics_resize "$METRICS_RESIZE" \
  --metrics_csv "$METRICS_CSV" \
  --exts "$EXTS" \
  "${EXTRA_ARGS[@]}"

echo "All done."
