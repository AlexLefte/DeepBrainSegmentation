import torch
import csv
from torch.utils.data import Dataset
import time
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    plt.bar(subjects_names, std_intensity_values, label='Std Intensity')
    plt.xlabel('Subjects')
    plt.ylabel('Intensity')
    plt.legend()
    plt.title('Comparison of MRI Intensity Across Subjects')
    plt.show()
#####################

