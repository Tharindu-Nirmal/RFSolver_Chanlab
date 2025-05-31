#!/bin/bash

# Define common options
PYTHON_SCRIPT="edit.py"
COMMON_ARGS="--num_steps 30 --name 'flux-dev' --offload"

# Define runs as individual strings (no line breaks inside array entries)
runs=(
"--source_prompt \"low resolution image\" --target_prompt \"high resolution image\" --guidance 2 --inject 10 --source_img_dir /scratch/gilbreth/lwickrem/data/HandpickedDegrads/superres/107.png"
"--source_prompt \"low resolution image\" --target_prompt \"high resolution image\" --guidance 2 --inject 10 --source_img_dir /scratch/gilbreth/lwickrem/data/HandpickedDegrads/superres/130014.jpg"
"--source_prompt \"low resolution image\" --target_prompt \"high resolution image\" --guidance 2 --inject 10 --source_img_dir /scratch/gilbreth/lwickrem/data/HandpickedDegrads/superres/134049.jpg"
"--source_prompt \"low resolution image\" --target_prompt \"high resolution image\" --guidance 2 --inject 10 --source_img_dir /scratch/gilbreth/lwickrem/data/HandpickedDegrads/superres/22.png"
"--source_prompt \"low resolution image\" --target_prompt \"high resolution image\" --guidance 2 --inject 10 --source_img_dir /scratch/gilbreth/lwickrem/data/HandpickedDegrads/superres/58_rain.png"
"--source_prompt \"low resolution image\" --target_prompt \"high resolution image\" --guidance 2 --inject 10 --source_img_dir /scratch/gilbreth/lwickrem/data/HandpickedDegrads/superres/city_read_03728.jpg"
)

# runs=(
# "--source_prompt \"\" --target_prompt \"\" --guidance 2 --inject 10 --source_img_dir /scratch/gilbreth/lwickrem/data/HandpickedDegrads/superres/107.png"
# "--source_prompt \"\" --target_prompt \"\" --guidance 2 --inject 10 --source_img_dir /scratch/gilbreth/lwickrem/data/HandpickedDegrads/superres/130014.jpg"
# "--source_prompt \"\" --target_prompt \"\" --guidance 2 --inject 10 --source_img_dir /scratch/gilbreth/lwickrem/data/HandpickedDegrads/superres/134049.jpg"
# "--source_prompt \"\" --target_prompt \"\" --guidance 2 --inject 10 --source_img_dir /scratch/gilbreth/lwickrem/data/HandpickedDegrads/superres/22.png"
# "--source_prompt \"\" --target_prompt \"\" --guidance 2 --inject 10 --source_img_dir /scratch/gilbreth/lwickrem/data/HandpickedDegrads/superres/58_rain.png"
# "--source_prompt \"\" --target_prompt \"\" --guidance 2 --inject 10 --source_img_dir /scratch/gilbreth/lwickrem/data/HandpickedDegrads/superres/city_read_03728.jpg"
# )

# Output directory base
OUTPUT_BASE="/scratch/gilbreth/lwickrem/RF_Inversion/RF-Solver-Edit/FLUX_Image_Edit/results/ddnm_superres_degrad_RFEdited"

# Run each experiment
for i in "${!runs[@]}"; do
  echo "Running experiment $((i+1))..."
  eval python $PYTHON_SCRIPT ${runs[$i]} $COMMON_ARGS --output_dir "$OUTPUT_BASE"
done
