import nibabel as nib
import numpy as np


def save_nifti(segmentation_volume, output_file):
    # Create a NIfTI image object
    nifti_img = nib.Nifti1Image(segmentation_volume, affine=np.eye(4))

    # Save the NIfTI image to a file
    nib.save(nifti_img, output_file)
