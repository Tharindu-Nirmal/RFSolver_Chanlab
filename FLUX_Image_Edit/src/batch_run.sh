#!/bin/bash

# Define common options
PYTHON_SCRIPT="edit_image.py"
COMMON_ARGS="--num_steps 30 --name 'flux-dev' --offload"

# Define runs as individual strings (no line breaks inside array entries)
# Single image tests for easy data
runs=(
"--source_prompt \"A black and white image of a cheetah. The fur of the leopard is bright, with dark spots. The leopard has dark streaks running down from its dark eyes.\" --target_prompt \"A colored image of a cheetah. The fur of the leopard is golden yellow, with dark spots. The leopard has dark streaks running down from its dark eyes.\" --guidance 4 --inject 4 --ddnm_inject 1 --degradation \"colorization\" --source_img_dir /scratch/gilbreth/lwickrem/data/afhq_degrads/color/cheetah_selected/flickr_wild_000557.jpg"
# "--source_prompt \"An black and white image of a man.\" --target_prompt \"A colorful image of a man. The man has black hair, and black eyes.\" --guidance 4 --inject 4 --ddnm_inject 1 --degradation \"colorization\" --source_img_dir /scratch/gilbreth/lwickrem/data/celeba_degrads/color/celeba_men_selected/30.jpg"
# "--source_prompt \"A low resolution image of a cat into the camera. The cat is white with black patches. The background is dark The nose of the cat is pink. The eyes of the cat are green.\" --target_prompt \"A high resolution image of a cat. The cat is white with black patches. The background is dark The nose of the cat is pink. The eyes of the cat are green.\" --guidance 4 --inject 8 --ddnm_inject 1 --degradation \"super resolution\" --source_img_dir /scratch/gilbreth/lwickrem/data/afhq_degrads/superres_8x/cat_selected/flickr_cat_000008.jpg"
# "--source_prompt \"A black and white image of a cat. The cat is white with black patches. The background is dark The nose of the cat is pink. The eyes of the cat are green.\" --target_prompt \"A colorful of a cat. The cat is white with black patches. The background is dark The nose of the cat is pink. The eyes of the cat are green.\" --guidance 4 --inject 4 --ddnm_inject 1 --degradation \"colorization\" --source_img_dir /scratch/gilbreth/lwickrem/data/afhq_degrads/color/cat_selected_1/flickr_cat_000008.jpg"
# "--source_prompt \"A blurred image of a cat. The cat is white with black patches. The background is dark.\" --target_prompt \"A sharp image of a cat. The cat has white fur with black spots. Highly detailed, taken using a Canon EOS R camera, hyper detailed photo-realistic maximum detail.\" --guidance 4 --inject 4 --ddnm_inject 1 --degradation \"deblurring\" --source_img_dir /scratch/gilbreth/lwickrem/data/afhq_degrads/deblur/cat_selected_1/flickr_cat_000008.jpg"
# "--source_prompt \"A noisy image of a cat. The cat is white with black patches. The background is dark.\" --target_prompt \"A clean, noise free image of a cat. The cat has white fur with black spots. There are no RGB noise artifacts on the output image. \" --guidance 4 --inject 4 --ddnm_inject 1 --degradation \"denoising\" --source_img_dir /scratch/gilbreth/lwickrem/data/afhq_degrads/denoise/cat_selected_1/flickr_cat_000008.jpg"
)

# PnP flow baseline
# runs=(
  # 4× SR example (use the HR image path here)
  # "--target_prompt \"A high-resolution, faithful reconstruction of the same cat.\" --guidance 4.0 --gamma 1 --eta_dn 0.3 --degradation \"super resolution\" --sr_scale 4 --source_img_dir /scratch/gilbreth/lwickrem/data/afhq_degrads/superres_4x/cat_selected/flickr_cat_000008.jpg" 
# )

# Single image tests for hard data
# runs=(
# "--source_prompt \"A black and white image of a leopard.\" --target_prompt \"A colorful image of a leopard. The tree is brown, with green leaves. The sky in the background is blue.\" --guidance 4 --inject 12 --ddnm_inject 1 --degradation \"colorization\" --source_img_dir /scratch/gilbreth/lwickrem/data/HandpickedDegrads/colorization/134049.jpg"
# "--source_prompt \"A black and white image of a giraffe.\" --target_prompt \"A colorful image of a giraffe.\" --guidance 4 --inject 1 --ddnm_inject 1 --degradation \"colorization\" --source_img_dir /scratch/gilbreth/lwickrem/data/HandpickedDegrads/colorization/130014.jpg"
# "--source_prompt \"A black and white image of a giraffe. The sky and trees are visible behind the giraffe. \" --target_prompt \"A colorful image of a giraffe. The sky and trees are visible behind the giraffe.\" --guidance 4 --inject 1 --ddnm_inject 1 --degradation \"colorization\" --source_img_dir /scratch/gilbreth/lwickrem/data/HandpickedDegrads/colorization/130014.jpg"
# "--source_prompt \"An image of a giraffe looking into the camera.\" --target_prompt \"A colorful image of a giraffe. The giraffe is brown and white. The sky is blue and the background has greenery.\" --guidance 4 --inject 10 --ddnm_inject 1 --degradation \"colorization\" --source_img_dir /scratch/gilbreth/lwickrem/data/HandpickedDegrads/colorization/130014.jpg"
# "--source_prompt \"An image of a giraffe looking into the camera.\" --target_prompt \"A colorful image of a giraffe. The giraffe is brown and white. The sky is blue and the background has greenery.\" --guidance 4 --inject 12 --ddnm_inject 1 --degradation \"colorization\" --source_img_dir /scratch/gilbreth/lwickrem/data/HandpickedDegrads/colorization/130014.jpg"
# )



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
OUTPUT_BASE="/scratch/gilbreth/lwickrem/RF_Inversion/RF-Solver-Edit/FLUX_Image_Edit/results/Colorize_ddnm_manyimgs"
# OUTPUT_BASE="/scratch/gilbreth/lwickrem/RF_Inversion/RF-Solver-Edit/FLUX_Image_Edit/results/Denoise_ddnm_manyimgs"
# OUTPUT_BASE="/scratch/gilbreth/lwickrem/RF_Inversion/RF-Solver-Edit/FLUX_Image_Edit/results/Superres_pnp_flow"
# OUTPUT_BASE="/scratch/gilbreth/lwickrem/RF_Inversion/RF-Solver-Edit/FLUX_Image_Edit/results/Colorize_ddnm_celebA_trials"

# Run each experiment
for i in "${!runs[@]}"; do
  echo "Running experiment $((i+1))..."
  eval python $PYTHON_SCRIPT ${runs[$i]} $COMMON_ARGS --output_dir "$OUTPUT_BASE"
done
