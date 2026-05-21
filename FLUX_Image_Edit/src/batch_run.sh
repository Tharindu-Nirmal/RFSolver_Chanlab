#!/bin/bash

# Define common options
PYTHON_SCRIPT="edit_image.py"
COMMON_ARGS="--num_steps 30 --name 'flux-dev' --offload"

# Define runs as individual strings (no line breaks inside array entries)
# Single image tests for easy data
# runs=(
# "--source_prompt \"A low resolution of a cheetah. The fur of the leopard is bright, with dark spots. The leopard has dark streaks running down from its dark eyes.\" --target_prompt \"A high resolution image of a cheetah. The fur of the leopard is golden yellow, with dark spots. The leopard has dark streaks running down from its dark eyes.\" --guidance 4 --inject 5 --ddnm_inject 1 --degradation \"super resolution\" --source_img_dir /scratch/gilbreth/lwickrem/data/LtF_test_degrads/superres_4x/pixabay_wild_000106.jpg"
# "--source_prompt \"An black and white image of a man.\" --target_prompt \"A colorful image of a man. The man has black hair, and black eyes.\" --guidance 4 --inject 4 --ddnm_inject 1 --degradation \"colorization\" --source_img_dir /scratch/gilbreth/lwickrem/data/celeba_degrads/color/celeba_men_selected/30.jpg"
# "--source_prompt \"A low resolution image of a cat into the camera. The cat is white with black patches. The background is dark The nose of the cat is pink. The eyes of the cat are green.\" --target_prompt \"A high resolution image of a cat. The cat is white with black patches. The background is dark The nose of the cat is pink. The eyes of the cat are green.\" --guidance 4 --inject 8 --ddnm_inject 1 --degradation \"super resolution\" --source_img_dir /scratch/gilbreth/lwickrem/data/afhq_degrads/superres_8x/cat_selected/flickr_cat_000008.jpg"
# )


# 4x superres
runs=(
"--source_prompt \"A low resolution of a cheetah. There are blocking artifacts on the image. \" --target_prompt \"A high resolution image of a cheetah. There are no blocking artifacts on the image. The image is smooth, and photorealistic. \" --guidance 4 --inject 5 --ddnm_inject 1 --degradation \"super resolution\" --source_img_dir /scratch/gilbreth/lwickrem/data/LtF_test_degrads/superres_4x/pixabay_wild_000106.jpg"
)


# For paper images
# runs=(
#   "--source_prompt \"A black and white image of a tiger. The tiger has fur with white and black streaks. The eyes of the tiger are dark. \" --target_prompt \"A colorful image of a tiger. The tiger has orange-golden fur with white and black streaks. The eyes of the tiger are dark orange.\" --guidance 2 --inject 10 --ddnm_inject 1 --degradation \"colorization\" --source_img_dir /scratch/gilbreth/lwickrem/data/LtF_test_degrads/color/flickr_wild_002809.jpg"
# )


# Single image tests for hard data
# runs=(
# "--source_prompt \"A black and white image of a leopard.\" --target_prompt \"A colorful image of a leopard. The tree is brown, with green leaves. The sky in the background is blue.\" --guidance 4 --inject 12 --ddnm_inject 1 --degradation \"colorization\" --source_img_dir /scratch/gilbreth/lwickrem/data/HandpickedDegrads/colorization/134049.jpg"
# )



# group runs for super-resolution
# runs=(
# "--source_prompt \"\" --target_prompt \"\" --guidance 2 --inject 10 --source_img_dir /scratch/gilbreth/lwickrem/data/HandpickedDegrads/superres/107.png"
# )

# "--source_prompt \"A low resolution image\" --target_prompt \"A high resolution image.\" --guidance 4 --inject 8 --source_img_dir /scratch/gilbreth/lwickrem/data/HandpickedDegrads/superres/107.png"

# "--source_prompt \"low resolution image\" --target_prompt \"A high resolution image.\" --guidance 2 --inject 10 --source_img_dir /scratch/gilbreth/lwickrem/data/HandpickedDegrads/superres/130014.jpg"
# )



# group runs for colorization
# runs=(
# "--source_prompt \"A black and white image of cars parked near a walkway.\" --target_prompt \"A colorful image of cars parked near a walkway. The cars are white. The walkway is grey.\" --guidance 4 --inject 12 --ddnm_inject 1 --degradation \"colorization\" --source_img_dir /scratch/gilbreth/lwickrem/data/HandpickedDegrads/colorization/107.png"
# )


# Output directory base
OUTPUT_BASE="/scratch/gilbreth/lwickrem/RF_Inversion/RF-Solver-Edit/FLUX_Image_Edit/results/Cleancheck/Superres4x"


# Run each experiment
for i in "${!runs[@]}"; do
  echo "Running experiment $((i+1))..."
  eval python $PYTHON_SCRIPT ${runs[$i]} $COMMON_ARGS --output_dir "$OUTPUT_BASE"
done
