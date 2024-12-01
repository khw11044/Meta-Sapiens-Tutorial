#!/bin/bash

# Kim Hyun Woo
# https://hueykim.github.io/


# Define folders and their respective model links
declare -A MODELS
MODELS=(
    ["pose"]="https://huggingface.co/facebook/sapiens-pose-0.3b-torchscript/resolve/main/sapiens_0.3b_goliath_best_goliath_AP_573_torchscript.pt2
              https://huggingface.co/facebook/sapiens-pose-0.6b-torchscript/resolve/main/sapiens_0.6b_goliath_best_goliath_AP_609_torchscript.pt2
              https://huggingface.co/facebook/sapiens-pose-1b-torchscript/resolve/main/sapiens_1b_goliath_best_goliath_AP_639_torchscript.pt2"
    ["seg"]="https://huggingface.co/facebook/sapiens-seg-0.3b-torchscript/resolve/main/sapiens_0.3b_goliath_best_goliath_mIoU_7673_epoch_194_torchscript.pt2
             https://huggingface.co/facebook/sapiens-seg-0.6b-torchscript/resolve/main/sapiens_0.6b_goliath_best_goliath_mIoU_7777_epoch_178_torchscript.pt2
             https://huggingface.co/facebook/sapiens-seg-1b-torchscript/resolve/main/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2"
    ["depth"]="https://huggingface.co/facebook/sapiens-depth-0.3b-torchscript/resolve/main/sapiens_0.3b_render_people_epoch_100_torchscript.pt2
               https://huggingface.co/facebook/sapiens-depth-0.6b-torchscript/resolve/main/sapiens_0.6b_render_people_epoch_70_torchscript.pt2
               https://huggingface.co/facebook/sapiens-depth-1b-torchscript/resolve/main/sapiens_1b_render_people_epoch_88_torchscript.pt2"
    ["normal"]="https://huggingface.co/facebook/sapiens-normal-0.3b-torchscript/resolve/main/sapiens_0.3b_normal_render_people_epoch_66_torchscript.pt2
                https://huggingface.co/facebook/sapiens-normal-0.6b-torchscript/resolve/main/sapiens_0.6b_normal_render_people_epoch_200_torchscript.pt2
                https://huggingface.co/facebook/sapiens-normal-1b-torchscript/resolve/main/sapiens_1b_normal_render_people_epoch_115_torchscript.pt2"
)

# Create checkpoints folder if it doesn't exist
mkdir -p checkpoints

# Loop through each folder and download the models
for folder in "${!MODELS[@]}"; do
    echo "Processing $folder models..."
    mkdir -p "checkpoints/$folder"
    
    model_sizes=("0.3b" "0.6b" "1b")
    index=0  # To map to model_sizes array

    for url in ${MODELS[$folder]}; do
        # Extract the original file name
        original_filename=$(basename "$url" | sed 's/?download=true//g')

        # Construct the new file name
        new_filename="sapiens_${model_sizes[$index]}_torchscript.pt2"
        index=$((index + 1))  # Increment index for the next model size

        # Set output paths
        output="checkpoints/$folder/$original_filename"
        renamed_output="checkpoints/$folder/$new_filename"

        echo "Downloading $original_filename..."
        wget --quiet --show-progress "$url" -O "$output" || { echo "Failed to download $url"; exit 1; }

        # Rename the file
        echo "Renaming $output to $renamed_output..."
        mv "$output" "$renamed_output"
    done
done

echo "All models downloaded and renamed successfully."
