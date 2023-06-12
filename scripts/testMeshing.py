import sys
sys.path.extend([ '/casa/install/python', '/usr/lib/python310.zip', '/usr/lib/python3.10', '/usr/lib/python3.10/lib-dynload', '/usr/local/lib/python3.10/dist-packages', '/usr/lib/python3/dist-packages', '/usr/lib/python3.10/dist-packages'])

import brainvisa.axon as axon
import brainvisa.processes as bv_proc

# Set the paths to the input and output files
input_file = "/mrhome/vladyslavz/git/central-sulcus-analysis/data/via11/nobackup/segm_results/skull_stripped_images/CS1x_noSST_noSynthAugm_monaiUnet/sub-via052/target.nii.gz"
output_dir = "/mrhome/vladyslavz/git/central-sulcus-analysis/data/via11/nobackup/segm_results/skull_stripped_images/CS1x_noSST_noSynthAugm_monaiUnet/sub-via052"

# Convert the binary image to a mesh surface
bv_proc.runProcess("topology.SurfaceFromMask",
                   input_file=input_file,
                   output_directory=output_dir,
                   output_mesh=True)

# Compute the sulcus depth profile
bv_proc.runProcess("sulci.SulcusDepthProfile",
                   input_mesh_file=f"{output_dir}/img.gii",
                   output_directory=output_dir)

# Load the sulcus depth profile as a NumPy array
depth_profile_file = f"{output_dir}/img_DepthProfile.txt"
depth_profile = axon.arrayFromFile(depth_profile_file)

# Print the sulcus depth profile
print(depth_profile)
