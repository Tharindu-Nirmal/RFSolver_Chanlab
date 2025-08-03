#!/bin/bash

# Define common options
PYTHON_SCRIPT="edit.py"
COMMON_ARGS="--num_steps 30 --name 'flux-dev' --offload"

# Define runs as individual strings (no line breaks inside array entries)
runs=(
"--source_prompt \"A black and white image of a leopard\" --target_prompt \"A colored image of a leopard.\" --guidance 2 --inject 8 --ddnm_inject 1 --degradation \"colorization\" --source_img_dir /scratch/gilbreth/lwickrem/data/HandpickedDegrads/colorization/134049.jpg"

# "--source_prompt \"A low resolution image\" --target_prompt \"A high resolution image.\" --guidance 4 --inject 8 --source_img_dir /scratch/gilbreth/lwickrem/data/HandpickedDegrads/superres/107.png"
# "--source_prompt \"A low resolution image\" --target_prompt \"A high resolution image.\" --guidance 6 --inject 8 --source_img_dir /scratch/gilbreth/lwickrem/data/HandpickedDegrads/superres/107.png"
# "--source_prompt \"A low resolution image\" --target_prompt \"A high resolution image of two parked cars near a walkway. The image is sharp and deblurred.\" --guidance 2 --inject 8 --source_img_dir /scratch/gilbreth/lwickrem/data/HandpickedDegrads/superres/107.png"
# "--source_prompt \"A low resolution image\" --target_prompt \"A high resolution image of two parked cars near a walkway. The image is sharp and deblurred.\" --guidance 4 --inject 8 --source_img_dir /scratch/gilbreth/lwickrem/data/HandpickedDegrads/superres/107.png"
# "--source_prompt \"A low resolution image of two parked cars near a walkway.\" --target_prompt \"A high resolution image of two parked cars near a walkway. The image is sharp and deblurred.\" --guidance 2 --inject 8 --source_img_dir /scratch/gilbreth/lwickrem/data/HandpickedDegrads/superres/107.png"

# "--source_prompt \"low resolution image\" --target_prompt \"A high resolution image.\" --guidance 2 --inject 10 --source_img_dir /scratch/gilbreth/lwickrem/data/HandpickedDegrads/superres/130014.jpg"
# "--source_prompt \"low resolution image\" --target_prompt \"A high resolution image.\" --guidance 2 --inject 10 --source_img_dir /scratch/gilbreth/lwickrem/data/HandpickedDegrads/superres/134049.jpg"
# "--source_prompt \"low resolution image\" --target_prompt \"A high resolution image.\" --guidance 2 --inject 10 --source_img_dir /scratch/gilbreth/lwickrem/data/HandpickedDegrads/superres/22.png"
# "--source_prompt \"low resolution image\" --target_prompt \"A high resolution image.\" --guidance 2 --inject 10 --source_img_dir /scratch/gilbreth/lwickrem/data/HandpickedDegrads/superres/573.png"
# "--source_prompt \"low resolution image\" --target_prompt \"A high resolution image.\" --guidance 2 --inject 10 --source_img_dir /scratch/gilbreth/lwickrem/data/HandpickedDegrads/superres/58_rain.png"
# "--source_prompt \"low resolution image\" --target_prompt \"A high resolution image.\" --guidance 2 --inject 10 --source_img_dir /scratch/gilbreth/lwickrem/data/HandpickedDegrads/superres/city_read_03728.jpg"
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
OUTPUT_BASE="/scratch/gilbreth/lwickrem/RF_Inversion/RF-Solver-Edit/FLUX_Image_Edit/results/Avgblur_RFedit_ddnm_trials"

# Run each experiment
for i in "${!runs[@]}"; do
  echo "Running experiment $((i+1))..."
  eval python $PYTHON_SCRIPT ${runs[$i]} $COMMON_ARGS --output_dir "$OUTPUT_BASE"
done
