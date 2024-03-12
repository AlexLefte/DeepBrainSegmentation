import nibabel as nib
import numpy as np
import os


def save_nifti(segmentation_volume, output_file):
    # Create a NIfTI image object
    nifti_img = nib.Nifti1Image(segmentation_volume, affine=np.eye(4))

    # Create the directory if it doesn't exist
    output_directory = os.path.dirname(output_file)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Save the NIfTI image to a file
    nib.save(nifti_img, output_file)
