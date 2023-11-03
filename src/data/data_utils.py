import torch
import csv
from torch.utils.data import Dataset
import time
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchio as tio
import nibabel.orientations as orientations
import os
from torchvision import transforms


# def compute_weights(labels: np.ndarray) -> np.ndarray:
#     """
#     Computes the classes weights matrix
#
#     Parameters
#     ----------
#     labels: np.ndarray
#         image labels
#     """
#     # Retrieve the unique classes found inside an image
#     # and also their frequency
#     unique, counts = np.unique(labels, return_counts=True)
#
#
#
#
#     # Median Frequency Balancing
#     class_wise_weights = np.median(counts) / counts
#     class_wise_weights[class_wise_weights > max_weight] = max_weight
#     (h, w, d) = mapped_aseg.shape
#
#     weights_mask = np.reshape(class_wise_weights[mapped_aseg.ravel()], (h, w, d))
#
#     return weights_mask


def get_labels(labels: np.ndarray,
               lut_labels: list,
               right_left_map: dict,
               plane: str,) -> np.ndarray:
    """
    Returns the labels in range: 0-95
    """
    # Process the labels: unknown => background
    # labels[labels not in lut_labels] = 0
    # labels = np.where(labels not in lut_labels, 0, labels)
    mask = ~np.isin(labels, lut_labels)
    # Use the mask to replace elements with 0
    labels[mask] = 0

    # Ensure there are no labels other than those listed in the lookup table
    assert all(item in labels for item in lut_labels), "Error: there are segmentation labels not listed in the LUT."

    if plane != "sagittal":
        # Create a new LUT into 0 - 95 range:
        lut_labels = {value: index for index, value in enumerate(lut_labels)}

        # Convert the original labels according to this LUT
        new_labels = np.vectorize(lut_labels.get)(labels)
        return new_labels
    else:
        # Process the labels based on the plane with the understanding that
        # the sagittal plane does not distinguish between hemispheres.

        # NOTE: if you want to lower the number of labels to 78, then uncomment
        # labels[labels >= 2000] -= 1000

        # For sagittal map all left structures to the right
        for right, left in right_left_map.items():
            labels[labels == left] = right

        # Create a new LUT for the sagittal labels
        lut_labels_sag = {value: index for index, value in enumerate(lut_labels)}
        # Convert the original labels according to this LUT
        new_labels = np.vectorize(lut_labels_sag.get)(labels)
        return new_labels


def get_labels_from_lut(path: str) -> dict.keys:
    """
    Get labels from LUT
    """
    return get_lut(path).keys()


def get_labels_from_lut(lut: pd.DataFrame) -> dict.keys:
    """
    Get labels from LUT
    """
    return lut["ID"]


def get_right_left_dict(lut: pd.DataFrame) -> dict:
    """
    Returns a dictionary that establishes mappings from structures
    in the Right Hemisphere to their corresponding structures in the Left Hemisphere.
    """
    # Initialize the dictionary
    right_left_dict = {}

    # Iterate through each structure of the dataframe
    for idx, name in zip(lut["ID"], lut["LabelName"]):
        if name.startswith("Right-"):
            # Get the name of the corresponding structure on left
            left_structure = "Left-" + name[6:]

            names = lut['LabelName']

            if lut['LabelName'].str.contains(left_structure).any():
                # Find the index of the right structure
                right_idx = [k for k, v in zip(lut["ID"], lut["LabelName"]) if v == left_structure][0]

                # Add the mapping to the left_to_right_map dictionary
                right_left_dict[idx] = right_idx
    return right_left_dict


def get_lut(path: str) -> pd.DataFrame:
    """
    Get the LUT in a data frame
    """
    # lut = {}
    #
    # # Read the Look-Up Table
    # with open(path, 'r') as file:
    #     for line in file:
    #         parts = line.strip().split('\t', 2)  # Use '\t' as the separator for TSV
    #         if len(parts) == 3:
    #             label_id, label_name, rgb = parts
    #             lut[0] = label_id
    #             lut[1] = [label_name, rgb]

    # Get the separator according to the file and read the tsv file
    separator = {"tsv": "\t", "csv": ",", "txt": " "}
    return pd.read_csv(path, sep=separator[path[-3:]])


def get_sagittal_labels_from_lut(lut: pd.DataFrame) -> list:
    """
    Returns the appropriate LUT labels for the sagittal plane
    """
    return [lut["ID"][index] for index, name in enumerate(lut["Name"]) \
            if not name.startswith("Left-") and not name.startswith("ctx-lh")]


# Data vizualization ###
def compare_intensity_across_dataset(subjects: list,
                                     subjects_names: list):
    # Initialize lists to store intensity statistics for each subject
    mean_intensity_values = []
    std_intensity_values = []

    # Calculate and compare intensity statistics for each subject and ROI
    for subject_data in subjects:
        # Print shapes
        print(subject_data.shape)

        # Calculate statistics
        mean_intensity = np.mean(subject_data)
        std_intensity = np.std(subject_data)

        # Append statistics to the lists
        mean_intensity_values.append(mean_intensity)
        std_intensity_values.append(std_intensity)

    print(mean_intensity_values)
    print(std_intensity_values)

    # Visualize and compare intensity values using plots or other methods
    plt.figure(figsize=(10, 5))
    plt.bar(subjects_names, mean_intensity_values, label='Mean Intensity')
    plt.xlabel('Subjects')
    plt.ylabel('Mean Intensity')
    plt.legend()
    plt.title('Comparison of MRI Mean Intensity Across Subjects')
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.bar(subjects_names, std_intensity_values, label='Std Intensity')
    plt.xlabel('Subjects')
    plt.ylabel('Std')
    plt.legend()
    plt.title('Comparison of MRI Standard Deviation Across Subjects')
    plt.show()
#####################


# Data Preprocessing
def crop_or_pad(image: np.ndarray, target_shape: list) -> np.ndarray:
    # Get the current shape of the data
    current_shape = list(image.shape)

    # Initialize padding values to zero
    padding = [(0, 0), (0, 0), (0, 0)]

    # Check if cropping or padding is needed in each dimension
    for i in range(len(current_shape)):
        if current_shape[i] < target_shape[i]:
            # If the current dimension is smaller than the target, add padding
            padding[i] = (0, target_shape[i] - current_shape[i])
        elif current_shape[i] > target_shape[i]:
            # If the current dimension is larger than the target, crop the data
            crop_amount = current_shape[i] - target_shape[i]
            # Calculate how much to crop from each side
            crop_start = crop_amount // 2
            crop_end = crop_amount - crop_start
            # Update the padding for this dimension
            padding[i] = (crop_start, crop_end)

    # Pad or crop the original data
    padded_data = np.pad(image, padding, mode='constant', constant_values=0)
    return padded_data


def fix_orientation(img, zooms, labels, plane: str = 'coronal') -> tuple:
    """
    Permutes the axis depending on the plane
    """
    # Specify the orientation that matches the desired anatomical orientation (e.g., RAS)
    desired_orientation = orientations.axcodes2ornt('RAS')

    # Apply the orientation to the image data
    img = nib.orientations.apply_orientation(img, desired_orientation)
    # zooms = nib.orientations.apply_orientation(zooms, desired_orientation)
    # labels = nib.orientations.apply_orientation(labels, desired_orientation)

    if plane == 'axial':
        img = np.moveaxis(img, [0, 1, 2], [1, 0, 2])
        # labels = np.moveaxis(labels, [0, 1, 2], [1, 2, 0])
        # weights = np.moveaxis(weights, [0, 1, 2], [1, 2, 0])
    elif plane == 'sagittal':
        img = np.moveaxis(img, [0, 1, 2], [2, 1, 0])
        # labels = np.moveaxis(labels, [0, 1, 2], [2, 1, 0])
        # weights = np.moveaxis(weights, [0, 1, 2], [2, 1, 0])
    return img, zooms, labels


def preprocess(image: np.ndarray, padding: int, mode: str) -> np.ndarray:
    """
    Performs preprocessing.
    There are several methods to preprocess the study:
    1) Crop or pad:
        - Set a standard input size.
        - If data shape is above the standard size => crop
        - If data shape is below the standard size => pad
    2) Normalization:
        * Intensity rescaling:
            - Use percentiles (e.g: (0.5, 99.5)).
            - Usually applied for CT scans
            - See U-net paper:  https://arxiv.org/pdf/1809.10486.pdf
        * Z-Score Normalization (in fact Standardization):
            - Can be applied after intensity rescaling
            - Can be applied after intensity rescaling
            - z = (x - mean) / std
        * Log normalization
            - Another approach would be to log the data and rescale using percentiles

    Attributes
    ----------
    image
        Unprocessed image as numpy.ndarray
    padding
        Padded input size to ensure the consistency
    mode
        Preprocessing mode:
        1) "percentiles_&_zscore"
        2) "log_norm_&_zscore"    """
    # 1) Crop or pad
    image = crop_or_pad(image, padding)
    #############

    # 2) Normalization using the torchio pipeline
    # Create transforms list
    transforms_list = []

    if mode == 'percentiles_&_zscore':
        # Append the RescaleIntensity and ZNormalization transforms
        transforms_list.append(tio.RescaleIntensity(percentiles=(0.5, 99.5)))
        transforms_list.append(tio.ZNormalization())
    elif mode == 'log_norm_&_zscore':
        # Append the ZNormalization transform
        transforms_list.append(tio.ZNormalization())
        # Apply log transform on the image
        # Note: TorchhIO does not implement such scaling
        image = np.log(image)
    else:
        # Wrong mode => return initial image
        print("Invalid mode.")
        return image

    # Initializing the torchio.Subject instance.
    # Create a subject
    subject = tio.Subject()

    # Create a TorchIO pipeline and add the transforms
    pipeline = tio.Compose(transforms_list)

    # Add the image to the subject
    scalar_image = tio.ScalarImage(tensor=image)
    subject.add_image(scalar_image, 'subject')

    # Apply the pipeline to the subject
    subject = pipeline(subject)

    return subject['subject']
####################

