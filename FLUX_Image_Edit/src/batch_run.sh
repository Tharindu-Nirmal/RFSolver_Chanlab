#!/bin/bash

# Define common options
PYTHON_SCRIPT="edit_image.py"
COMMON_ARGS="--num_steps 30 --name 'flux-dev' --offload"

# Runs one image for each type of degradation. Edit these to run different images or change the prompts, guidance, inject, etc. parameters. The source image is specified in the --source_img_dir argument. The output will be saved in the directory specified by --output_dir, with a subdirectory for each run.  
runs=(
"--source_prompt \"A low resolution of a cheetah. There are blocking artifacts on the image. \" --target_prompt \"A high resolution image of a cheetah. There are no blocking artifacts on the image. The image is smooth, and photorealistic. \" --degradation \"super resolution\" --guidance 4 --inject 5 --lambda_start 0.50 --lambda_step 0.70 --lambda_end 0.85 --lambda_level_hi 1.0 --lambda_level_lo 0.5 --lambda_final_pad 3 --source_img_dir /scratch/gilbreth/lwickrem/data/LtF_test_degrads/superres_4x/pixabay_wild_000106.jpg"
"--source_prompt \"A black and white image of a cheetah. The fur of the cheetah is bright, with dark spots. The cheetah has dark streaks running down from his dark eyes.\" --target_prompt \"A colored image of a cheetah. The fur of the cheetah is golden yellow, with dark spots. The cheetah has dark streaks running down from his dark eyes.\" --degradation \"colorization\" --guidance 4 --inject 5 --lambda_start 0.40 --lambda_step 0.50 --lambda_end 0.95 --lambda_level_hi 1.0 --lambda_level_lo 0.8 --lambda_final_pad 1 --source_img_dir /scratch/gilbreth/lwickrem/data/LtF_test_degrads/color/pixabay_wild_000106.jpg"
"--source_prompt \"A noisy image of a cheetah. The fur of the cheetah is bright, with dark spots. The cheetah has dark streaks running down from his dark eyes.\" --target_prompt \"A clean, noise free image of a cheetah. The fur of the cheetah is bright, with dark spots. The cheetah has dark streaks running down from his dark eyes. Highly detailed, taken using a Canon EOS R camera, hyper detailed photo-realistic maximum detail. There are no noise artifacts.\" --degradation \"denoising\" --guidance 4 --inject 5 --lambda_start 0.50 --lambda_step 0.75 --lambda_end 0.95 --lambda_level_hi 1.0 --lambda_level_lo 0.5 --lambda_final_pad 2 --source_img_dir /scratch/gilbreth/lwickrem/data/LtF_test_degrads/denoise/pixabay_wild_000106.jpg"
"--source_prompt \"A blurred image of a cheetah. The fur of the cheetah is bright, with dark spots. The cheetah has dark streaks running down from his dark eyes.\" --target_prompt \"A sharp image of a cheetah. The fur of the cheetah is bright, with dark spots. The cheetah has dark streaks running down from his dark eyes. Highly detailed, taken using a Canon EOS R camera, hyper detailed photo-realistic maximum detail.\" --degradation \"deblurring\" --guidance 4 --inject 5 --lambda_start 0.70 --lambda_step 0.80 --lambda_end 0.90 --lambda_level_hi 1.0 --lambda_level_lo 0.3 --lambda_final_pad 3 --source_img_dir /scratch/gilbreth/lwickrem/data/LtF_test_degrads/deblur/pixabay_wild_000106.jpg"
)


# Output directory base
OUTPUT_BASE="/scratch/gilbreth/lwickrem/RF_Inversion/RF-Solver-Edit/FLUX_Image_Edit/results/Cleancheck/alldegrads_cheetah"


# Run each experiment
for i in "${!runs[@]}"; do
  echo "Running experiment $((i+1))..."
  eval python $PYTHON_SCRIPT ${runs[$i]} $COMMON_ARGS --output_dir "$OUTPUT_BASE"
done
