#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Usage:
#   bash run_folder.sh /path/to/input_imgs /path/to/output_base
# -----------------------------

# Positional args (with sane fallbacks for testing)
INPUT_DIR="${1:-/scratch/gilbreth/lwickrem/data/afhq_degrads/color/cat_selected}"
OUTPUT_BASE="${2:-/scratch/gilbreth/lwickrem/RF_Inversion/RF-Solver-Edit/FLUX_Image_Edit/results/Colorize_ddnm_cat_selected_1}"

# Script + common args (use an array to avoid eval/quoting issues)
PYTHON_SCRIPT="edit.py"
COMMON_ARGS=(--num_steps 30 --name flux-dev --offload)

# -----------------------------
# Hyperparameters (same for all images)
# -----------------------------
SOURCE_PROMPT='An image of a cat.'
TARGET_PROMPT='A colorful image of a cat. The background is dark. The nose of the cat is pink. The eyes of the cat are green.'
GUIDANCE=4
INJECT=8
DDNM_INJECT=1
DEGRADATION='colorization'

# -----------------------------
# Options
# -----------------------------
# Recurse through subfolders?
RECURSIVE=false   # set to true if you want to walk subdirs
# Save each image’s output in its own subfolder under OUTPUT_BASE?
SEPARATE_SUBDIRS=true
# File extensions to include
EXTS=(jpg jpeg png bmp webp tif tiff)

mkdir -p "$OUTPUT_BASE"

# Build a find expression for extensions (case-insensitive)
build_find_ext_expr() {
  local expr=""
  for ext in "${EXTS[@]}"; do
    if [[ -z "$expr" ]]; then
      expr="-iname *.$ext"
    else
      expr="$expr -o -iname *.$ext"
    fi
  done
  printf "%s" "$expr"
}

process_one() {
  local img="$1"

  # Choose output directory style
  local outdir="$OUTPUT_BASE"
  if [[ "$SEPARATE_SUBDIRS" == "true" ]]; then
    local stem
    stem="$(basename "${img%.*}")"
    outdir="$OUTPUT_BASE/$stem"
  fi
  mkdir -p "$outdir"

  echo "→ Processing: $img"
  echo "  Output dir: $outdir"

  # Optional: skip if outdir already has files (uncomment to enable skipping)
  # if compgen -G "$outdir/*" > /dev/null; then
  #   echo "  Skipping (already populated): $outdir"
  #   return 0
  # fi

  python "$PYTHON_SCRIPT" \
    --source_prompt "$SOURCE_PROMPT" \
    --target_prompt "$TARGET_PROMPT" \
    --guidance "$GUIDANCE" \
    --inject "$INJECT" \
    --ddnm_inject "$DDNM_INJECT" \
    --degradation "$DEGRADATION" \
    --source_img_dir "$img" \
    "${COMMON_ARGS[@]}" \
    --compute_metrics \
    --id_metric clip \
    --metrics_csv "$OUTPUT_BASE/metrics.csv" \
    --output_dir "$outdir"
}

# Gather and process files
if [[ "$RECURSIVE" == "true" ]]; then
  # Recursive: robust to spaces via -print0
  ext_expr=$(build_find_ext_expr)
  # shellcheck disable=SC2086
  while IFS= read -r -d '' img; do
    process_one "$img"
  done < <(find "$INPUT_DIR" -type f \( $ext_expr \) -print0)
else
  # Non-recursive: simple globbing per extension
  shopt -s nullglob
  for ext in "${EXTS[@]}"; do
    for img in "$INPUT_DIR"/*."$ext"; do
      process_one "$img"
    done
    # also handle uppercase extensions
    for img in "$INPUT_DIR"/*."${ext^^}"; do
      process_one "$img"
    done
  done
fi

echo "All done."
