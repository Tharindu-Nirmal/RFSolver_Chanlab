#!/bin/bash

# Define common options
PYTHON_SCRIPT="edit.py"
COMMON_ARGS="--num_steps 30 --name 'flux-dev' --offload"

# Define runs as individual strings (no line breaks inside array entries)
# Single image tests
runs=(
# "--source_prompt \"A black and white image of a leopard\" --target_prompt \"A colorful image of a leopard. The tree is brown, with green leaves. The sky in the background is blue.\" --guidance 4 --inject 12 --ddnm_inject 1 --degradation \"colorization\" --source_img_dir /scratch/gilbreth/lwickrem/data/HandpickedDegrads/colorization/134049.jpg"
"--source_prompt \"A black and white image of a giraffe.\" --target_prompt \"A colorful image of a giraffe. The giraffe is brown and white. The sky is blue and the background has greenery.\" --guidance 4 --inject 12 --ddnm_inject 1 --degradation \"colorization\" --source_img_dir /scratch/gilbreth/lwickrem/data/HandpickedDegrads/colorization/130014.jpg"
)


# group runs for super-resolution
# runs=(
# "--source_prompt \"\" --target_prompt \"\" --guidance 2 --inject 10 --source_img_dir /scratch/gilbreth/lwickrem/data/HandpickedDegrads/superres/107.png"
# "--source_prompt \"\" --target_prompt \"\" --guidance 2 --inject 10 --source_img_dir /scratch/gilbreth/lwickrem/data/HandpickedDegrads/superres/130014.jpg"
# "--source_prompt \"\" --target_prompt \"\" --guidance 2 --inject 10 --source_img_dir /scratch/gilbreth/lwickrem/data/HandpickedDegrads/superres/134049.jpg"
# "--source_prompt \"\" --target_prompt \"\" --guidance 2 --inject 10 --source_img_dir /scratch/gilbreth/lwickrem/data/HandpickedDegrads/superres/22.png"
# "--source_prompt \"\" --target_prompt \"\" --guidance 2 --inject 10 --source_img_dir /scratch/gilbreth/lwickrem/data/HandpickedDegrads/superres/58_rain.png"
# "--source_prompt \"\" --target_prompt \"\" --guidance 2 --inject 10 --source_img_dir /scratch/gilbreth/lwickrem/data/HandpickedDegrads/superres/city_read_03728.jpg"
# )

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
# )



# group runs for colorization
# runs=(
# "--source_prompt \"A black and white image of cars parked near a walkway.\" --target_prompt \"A colorful image of cars parked near a walkway. The cars are white. The walkway is grey.\" --guidance 4 --inject 12 --ddnm_inject 1 --degradation \"colorization\" --source_img_dir /scratch/gilbreth/lwickrem/data/HandpickedDegrads/colorization/107.png"
# "--source_prompt \"A black and white image of a giraffe.\" --target_prompt \"A colorful image of a giraffe. The giraffe is brown and white. The sky is blue and the background has greenery.\" --guidance 4 --inject 12 --ddnm_inject 1 --degradation \"colorization\" --source_img_dir /scratch/gilbreth/lwickrem/data/HandpickedDegrads/colorization/130014.jpg"
# "--source_prompt \"A black and white image of a leopard\" --target_prompt \"A colorful image of a leopard. The tree is brown, with green leaves. The sky in the background is blue.\" --guidance 4 --inject 12 --ddnm_inject 1 --degradation \"colorization\" --source_img_dir /scratch/gilbreth/lwickrem/data/HandpickedDegrads/colorization/134049.jpg"
# "--source_prompt \"A black and white image of a smiling girl\" --target_prompt \"A colorful image of a smiling girl. She is wearing a blue jacket. The background is dark blue.\" --guidance 4 --inject 12 --ddnm_inject 1 --degradation \"colorization\" --source_img_dir /scratch/gilbreth/lwickrem/data/HandpickedDegrads/colorization/22.png"
# "--source_prompt \"A black and white image of cars parked in a city.\" --target_prompt \"A colorful image of cars parked in a city. The buildings in the background a are white, and the sky is blue.\" --guidance 4 --inject 12 --ddnm_inject 1 --degradation \"colorization\" --source_img_dir /scratch/gilbreth/lwickrem/data/HandpickedDegrads/colorization/573.png"
# "--source_prompt \"A black and white image of cars parked near a road.\" --target_prompt \"A colorful image of cars parked near a road. The sky is blue and the trees have green leaves.\" --guidance 4 --inject 12 --ddnm_inject 1 --degradation \"colorization\" --source_img_dir /scratch/gilbreth/lwickrem/data/HandpickedDegrads/colorization/58_rain.png"
# "--source_prompt \"A black and white image of mercedes car on a city street\" --target_prompt \"A colorful image of a purple mercedes car on a city street. The buildings on the background are light yellow. \" --guidance 4 --inject 12 --ddnm_inject 1 --degradation \"colorization\" --source_img_dir /scratch/gilbreth/lwickrem/data/HandpickedDegrads/colorization/city_read_03728.jpg"
# )


# Output directory base
# OUTPUT_BASE="/scratch/gilbreth/lwickrem/RF_Inversion/RF-Solver-Edit/FLUX_Image_Edit/results/Colorize_ddnm_manyimgs"
OUTPUT_BASE="/scratch/gilbreth/lwickrem/RF_Inversion/RF-Solver-Edit/FLUX_Image_Edit/results/Colorize_ddnm_trials"

# Run each experiment
for i in "${!runs[@]}"; do
  echo "Running experiment $((i+1))..."
  eval python $PYTHON_SCRIPT ${runs[$i]} $COMMON_ARGS --output_dir "$OUTPUT_BASE"
done
