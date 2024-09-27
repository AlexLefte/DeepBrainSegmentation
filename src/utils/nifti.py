import nibabel as nib
import numpy as np
import os


def save_nifti(segmentation_volume, output_file, affine=None):
    # Create a NIfTI image object
    if affine is None:
        affine = np.eye(4)
    nifti_img = nib.Nifti1Image(segmentation_volume, affine=affine)

    # Create the directory if it doesn't exist
    output_directory = os.path.dirname(output_file)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Save the NIfTI image to a file
    nib.save(nifti_img, output_file)
