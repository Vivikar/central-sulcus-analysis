#!/bin/bash

# Directory to search for input files
input_dir="/mrhome/vladyslavz/git/SynthSeg/data/training_label_maps"

# Count the number of input files
num_files=$(find "$input_dir" -type f -name "csf_mask*.nii.gz" | wc -l)

# Loop through all files with a .nii.gz extension
counter=1
find "$input_dir" -type f -name "csf_mask*.nii.gz" | while read -r input_file; do

  # Generate the output file name
  output_file="${input_file/csf_mask/skeleton_csf}"

  # Run the VipSkeleton command
  VipSkeleton -i "$input_file" -so "$output_file" -sk s -v n
  echo $output_file
  # Update the progress bar
  progress=$(echo "scale=2; $counter/$num_files" | bc)
  printf "\rProcessing file %s of %s (%.2f%%)" "$counter" "$num_files" "$(echo "$progress*100" | bc)"
  ((counter++))
# break
done

# Print a newline after the progress bar
printf "\n"
