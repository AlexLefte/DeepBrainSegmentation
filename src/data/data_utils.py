import nibabel as nib
import nibabel.processing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.utils.data
import torchio as tio
import nibabel.orientations as orientations
from scipy.ndimage import affine_transform
from scipy.ndimage import binary_opening
from scipy.ndimage import binary_closing
from numpy.linalg import inv
import logging
import random
from skimage import measure
import os

ORIG = 't1weighted.nii.gz'
LABELS = 'labels.DKT31.manual+aseg.nii.gz'
LOGGER = logging.getLogger(__name__)


# region Class weights
# def compute_weights(labels: list,
#                     loss_fn: str) -> (list, dict):
#     """
#     Computes the classes weights matrix
#
#     Parameters
#     ----------
#     labels: list
#         Label arrays list
#     loss_fn: str
#         Loss function type
#     """
#     # Stack the labelled slices:
#     stacked_labels = np.stack(labels, axis=0)
#
#     # Get the unique values in the label matrix (sorted asc)
#     class_names, class_count = np.unique(stacked_labels, return_counts=True)
#
#     # Convert to float type
#     class_count = np.array(class_count, dtype=float)
#
#     # Create the weights dictionary
#     weights_dict = dict(zip(class_names, class_count))
#
#     # Compute the class weights according to the loss function in use
#     if loss_fn == 'unified_focal_loss':
#         # Class weights: within the [0, 1] range
#         # Bkg < 0.5, Fg -> [0.6, 1]
#         weights_dict = compute_non_linear_weights(weights_dict)
#     elif loss_fn == 'dice_loss_&_cross_entropy':
#         # Compute the median frequency balanced weights
#         weights_dict = compute_median_frequency_balanced_weights(weights_dict)
#
#     # Get the weights arrays
#     weights_list = get_weights_list(labels, weights_dict)
#
#     # Return the dictionary
#     return weights_list, weights_dict

def compute_weights(labels: np.ndarray,
                    loss_fn: str) -> (list, dict):
    """
    Computes the classes weights matrix

    Parameters
    ----------
    labels: list
        Label arrays list
    loss_fn: str
        Loss function type
    """
    # Get the unique values in the label matrix (sorted asc)
    class_names, class_count = np.unique(labels, return_counts=True)

    # Convert to float type
    class_count = np.array(class_count, dtype=float)

    # Create the weights dictionary
    weights_dict = dict(zip(class_names, class_count))

    # Compute the class weights according to the loss function in use
    if loss_fn == 'unified_focal_loss':
        # Class weights: within the [0, 1] range
        # Bkg < 0.5, Fg -> [0.6, 1]
        weights_dict = compute_non_linear_weights(weights_dict)
    elif loss_fn == 'dice_loss_&_cross_entropy':
        # Compute the median frequency balanced weights
        weights_dict = compute_median_frequency_balanced_weights(weights_dict)

    # Get the weights arrays
    weights = get_weights_list(labels, weights_dict)

    # Return the dictionary
    return weights, list(weights_dict.values())


# def get_weights_list(labels: list,
#                      weights_dict: dict):
#     # Initialize the weights list
#     weights_list = []
#
#     for label_array in labels:
#         # Apply the mapping to the original matrix
#         weights_array = np.vectorize(weights_dict.get)(label_array)
#
#         # Append the new weight matrix to the collection:
#         weights_list.append(weights_array)
#
#     # Return the list of 2D arrays
#     return weights_list

def get_weights_list(labels: np.ndarray,
                     weights_dict: dict):
    weights = np.vectorize(weights_dict.get)(labels)

    # Return the list of 2D arrays
    return weights


def get_weights_list_from_mask(labels: torch.Tensor,
                               weights_mask: torch.Tensor,
                               num_classes: int):
    """
    Returns a list of weights for each class based on the class labels and weights mask.

    Parameters
    ----------
    labels : torch.Tensor
        Tensor of class labels.

    weights_mask : torch.Tensor
        Tensor of weights corresponding to the class labels.

    num_classes : int
        Total number of classes.

    Returns
    -------
    torch.Tensor
        Tensor of weights for each class.
    """
    # Initialize the class weights list with zeros
    class_weights = np.zeros(num_classes)

    # Flatten the labels and weights_mask for vectorized operations
    labels_flat = labels.flatten()
    weights_flat = weights_mask.flatten()

    # Find unique classes in the labels and their first occurrence indices
    unique_classes, indices = np.unique(labels_flat, return_index=True)

    # Assign the weight for each class, if present in labels
    for class_idx in unique_classes:
        if class_idx < num_classes:  # Ensure the class index is within the valid range
            # Get the index of the first occurrence
            first_occurrence_idx = indices[unique_classes == class_idx][0]
            class_weights[class_idx] = weights_flat[first_occurrence_idx]

    # Convert the weights list to a PyTorch tensor
    return torch.tensor(class_weights)


def compute_median_frequency_balanced_weights(class_weights: dict):
    """
    Computes median frequency balanced weights
    """
    # Compute the median
    median_count = np.median(list(class_weights.values()))

    # Compute each weight
    for label, count in class_weights.items():
        class_weights[label] = float(median_count) / class_weights[label]

    # Return the dictionary
    return class_weights


def compute_non_linear_weights(class_weights: dict,
                               r: float = 1.5):
    """
    Computes non-linear weights for each class based on their frequencies.

    Parameters
    ----------
    class_weights : dict
        Dictionary where keys are class labels and values are initial class frequencies.
    r : float
        Parameter controlling the non-linear transformation applied to class weights.

    Returns
    -------
    dict
        Dictionary with updated class weights.
    """
    # Step 1: Invert the frequencies for all classes, except for the background class
    class_freqs = list(class_weights.values())[1:]
    class_freqs = [float(1 / c) for c in class_freqs]
    for label, weight in class_weights.items():
        class_weights[label] = 1 / class_weights[label]

    # Step 2: Apply a non-linear transformation to foreground class weights.
    fg_min_weight = min(list(class_weights.values())[1:])
    fg_max_weight = max(list(class_weights.values())[1:])

    for label, weight in class_weights.items():
        if label != 0:
            class_weights[label] = fg_max_weight - (fg_max_weight - fg_min_weight) * (
                        ((fg_max_weight - class_weights[label]) / (fg_max_weight - fg_min_weight)) ** r)
        else:
            class_weights[0] = 0.1

    # Step 3: Normalize transformed weights to the range [0.6, 0.8].
    min_transformed_weight = min(list(class_weights.values())[1:])
    max_transformed_weight = max(list(class_weights.values())[1:])

    for label, weight in class_weights.items():
        if label != 0:
            class_weights[label] = 0.6 + 0.2 * ((class_weights[label] - min_transformed_weight) / (
                    max_transformed_weight - min_transformed_weight))

    x = np.linspace(min(class_freqs), max(class_freqs), 10000)  # 500 de puncte între 1 și 100

    # Calculăm class_weights[label] pentru fiecare x
    b = np.zeros_like(x)
    for i, x_temp in enumerate(x):
        if x_temp < fg_min_weight:
            b[i] = fg_min_weight * ((x_temp / fg_min_weight) ** r)
        else:
            b[i] = fg_max_weight - (fg_max_weight - fg_min_weight) * (
                        ((fg_max_weight - x_temp) / (fg_max_weight - fg_min_weight)) ** r)

    # Calculăm valorile finale pentru plotare
    min_val = min(b)
    max_val = max(b)
    final_values = 0.6 + 0.3 * ((b - min(b)) / (
            max(b) - min(b)))

    to_plot = {}
    for i, f in enumerate(class_freqs):
        to_plot[f] = class_weights[i + 1]

    # plt.figure(figsize=(10, 5))
    # plt.plot(x, final_values, zorder=1)
    # plt.scatter(to_plot.keys(), to_plot.values(), color='red', label='wi',
    #             zorder=2)  # Scatter plot of dictionary values
    # plt.title('wi = f(1 / fi)')
    # plt.xlabel('1 / fi')
    # plt.ylabel('wi')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # Return the updated class weights.
    return class_weights


def normalize_weights(weights_list: np.ndarray,
                      eps: int = 0.000001):
    """
    Normalizes the weights within the [0, 1] range

    Parameters
    ----------
    weights_list: np.ndarray
        Un-normalized class weights
    eps: float
        Minimum weight in order to avoid a zero weight for background
    """
    # Compute the minimum and maximum weight
    min_weight = np.min(weights_list)
    max_weight = np.max(weights_list)

    # Compute the new weight for each class
    for i in range(len(weights_list)):
        weights_list[i] = (weights_list[i] + eps - min_weight) / (max_weight - min_weight + eps)

    # Return the result
    return weights_list


# endregion


# region Labels Processing
def lut2labels(labels: np.ndarray,
               lut_labels: list,
               right_left_map: dict,
               plane: str,
               unilat: bool = False) -> np.ndarray:
    """
    Returns the labels in range: 0-78
    """
    # Delateralize cortical structures between 2000 & 2099 (that are not included in the LUT)
    delat_structures = [x for x in range(2000, 2036) if x not in lut_labels]
    mask = np.isin(labels, delat_structures)
    labels[mask] -= 1000

    if unilat or plane == 'sagittal':
        # For sagittal map all right structures to the left
        for right, left in right_left_map.items():
            labels[labels == right] = left

    # Process the labels: unknown => background
    mask = ~np.isin(labels, lut_labels)
    # Use the mask to replace elements with 0
    labels[mask] = 0

    # Ensure there are no labels other than those listed in the lookup table
    assert all(item in labels for item in lut_labels), "Error: there are segmentation labels not listed in the LUT."

    # Create a new LUT into 0-78 range (0-51 for the sagittal plane):
    lut_labels = {value: index for index, value in enumerate(lut_labels)}

    # Convert the original labels according to this LUT
    new_labels = np.vectorize(lut_labels.get)(labels)
    new_labels = np.asarray(new_labels, dtype=np.uint8)
    return new_labels


def labels2lut(labels: np.ndarray,
               lut_labels_dict: dict):
    """
    Converts the label IDs back to the initial LUT IDs
    """
    reverted_dict = {value: key for key, value in lut_labels_dict.items()}
    return np.vectorize(reverted_dict.get)(labels)


def get_labels_from_lut(lut: pd.DataFrame) -> dict.keys:
    """
    Get labels from LUT
    """
    return lut["ID"]


def get_lut_from_labels(labels,
                        lut_labels):
    # Create a mapping dictionary
    mapping_dict = {i: float(lut_labels[i]) for i in range(len(lut_labels))}

    # Map the array values to labels using the dictionary
    mapped_labels = np.vectorize(mapping_dict.get)(labels)

    # Return
    return mapped_labels


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
        # elif name.startswith("ctx-rh-"):
        #     # Get the name of the corresponding structure on left
        #     left_structure = "ctx-lh-" + name[7:]
        else:
            continue

        # Search the index of the left-side structure
        if lut['LabelName'].str.contains(left_structure).any():
            # Find the index of the right structure
            left_idx = [k for k, v in zip(lut["ID"], lut["LabelName"]) if v == left_structure][0]

            # Add the mapping to the left_to_right_map dictionary
            right_left_dict[idx] = left_idx
    return right_left_dict


def get_lut(path: str) -> pd.DataFrame:
    """
    Get the LUT in a data frame
    """
    # Get the separator according to the file and read the tsv file
    separator = {"tsv": "\t", "csv": ",", "txt": " "}
    return pd.read_csv(path, sep=separator[path[-3:]])


def get_sagittal_labels_from_lut(lut: pd.DataFrame) -> list:
    """
    Returns the appropriate LUT labels for the sagittal plane
    """
    return [lut["ID"][index] for index, name in enumerate(lut["LabelName"])
            if not name.startswith("Right-") and not name.startswith("ctx-rh")]


# endregion


# region Data visualization
def compare_intensity_across_subjects(subjects: list,
                                      subjects_names: list):
    # Initialize lists to store intensity statistics for each subject
    mean_intensity_values = []
    std_intensity_values = []

    # Calculate and compare intensity statistics for each subject and ROI
    for subject_data in subjects:
        # Print shapes
        # print(subject_data.shape)

        # Calculate statistics
        mean_intensity = np.mean(subject_data)
        std_intensity = np.std(subject_data)

        # Append statistics to the lists
        mean_intensity_values.append(mean_intensity)
        std_intensity_values.append(std_intensity)

    # print(mean_intensity_values)
    # print(std_intensity_values)

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


def compare_intensity_across_dataset(stacked_slices: np.ndarray):
    # Stack the slices along a new axis (axis=0)
    # stacked_arrays = np.stack(slices, axis=0)

    # Compute mean and std along the new axis
    mean_intensity = np.mean(stacked_slices)
    std_intensity = np.std(stacked_slices)

    # Compute min, max values
    min_intensity = np.min(stacked_slices)
    max_intensity = np.max(stacked_slices)

    print("=====================================")
    print(f"Mean along dataset: {mean_intensity}")
    print(f"Std along dataset: {std_intensity}")
    print(f"Min value: {min_intensity}")
    print(f"Max value: {max_intensity}")
    print("======================================")


def compare_intensity(original,
                      processed):
    # Calculate statistics
    print("Original mean: " + str(np.mean(original)))
    print("Original std: " + str(np.std(original)))
    print("Processed mean: " + str(np.mean(processed)))
    print("Processed std: " + str(np.std(processed)))


def plot_histogram(data: np.ndarray,
                   title: str = ''):
    # Get stacked data:
    stacked_data = np.stack(data, axis=0)

    # Flatten the stacked data to create a 1D array
    flatten_data = stacked_data.flatten()

    # Plot the histogram
    plt.hist(flatten_data, bins=100, edgecolor='black')  # Adjust the number of bins as needed
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # Show the plot
    plt.show()


def plot_loss_curves(results: dict[str, list[float]]):
    """
    Plots training curves of a results dictionary
    """
    # Get the loss values of the results dictionary (training and test)
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    # Figure out how many epochs there were
    epochs = range(len(results["train_loss"]))

    # Setup a plot
    plt.figure(figsize=(15, 7))

    # Plot the loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    # plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Combined Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot the accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    # plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()


def plot_slices(slices: list, title: str = ''):
    """
    Plots a series of slices.

    Parameters:
    - slices: MRI slices to plot
    - title: Can be the subject's name, or any kind of information
    """
    # Plot multiple slices along the first axis
    num_slices = len(slices)
    fig, axes = plt.subplots(1, num_slices, figsize=(15, 5))

    for i in range(num_slices):
        axes[i].imshow(slices[i], cmap='gray', origin='lower')
        axes[i].set_title(f'Slice {i}')
        axes[i].axis('off')
    plt.show()


# endregion


# region Loading and Preprocessing Data
def remove_blank_slices(images: np.ndarray,
                        labels: np.ndarray,
                        weights: np.ndarray,
                        threshold: int = 5):
    """
    Removes slices with very few labeled voxels.

    Parameters
    ----------
    images: ndarray
        the MRI volume as a
    labels: ndarray
        the labeled volume
    weights: ndarray
        the weights masks
    threshold:
        the minimum number of foreground pixels a slice has to accomplish in order to be kept

    Returns
    -------
    The volumes without those slices that do not meet the requirements
    """
    # Compute the sums of the labels for each slice
    slices_nonzero_counts = np.count_nonzero(labels, axis=(1, 2))

    # Select those slices with at least 20 voxels different from background
    selected_slices = np.where(slices_nonzero_counts > threshold)

    # Return the selected slices
    return images[selected_slices], labels[selected_slices], weights[selected_slices]


def crop_or_pad(image: np.ndarray, labels: np.ndarray, target_shape: list) -> (np.ndarray, np.ndarray):
    """
    Crops or pads slices to fit the standard dimension given by the configuration file.
    """
    # Assert shapes match
    assert image.shape == labels.shape, ("The original image and its associated labels"
                                         " have different shapes")

    # Get the current shape of the data
    current_shape = image.shape

    # Initialize padding values to zero
    padding = [(0, 0), (0, 0), (0, 0)]

    # Initialize the final images
    orig = np.zeros(target_shape)
    label = np.zeros(target_shape)

    # Iterate over each dimension
    for i in range(3):
        if current_shape[i] < target_shape[i]:
            # If the current dimension is smaller than the target, add padding
            pad_amount = target_shape[i] - current_shape[i]
            pad_start = pad_amount // 2
            pad_end = pad_amount - pad_start
            padding[i] = (pad_start, pad_end)
        elif current_shape[i] > target_shape[i]:
            # If the current dimension is larger than the target, crop the data
            crop_amount = current_shape[i] - target_shape[i]
            crop_start = crop_amount // 2
            crop_end = current_shape[i] - (crop_amount - crop_start)
            padding[i] = (crop_start, crop_end)

    # Pad or crop the original data
    orig = np.pad(image, padding, mode='constant', constant_values=0)
    label = np.pad(labels, padding, mode='constant', constant_values=0)

    # # Pad or crop on x-axis
    # if current_shape[0] < target_shape[0]:
    #     # If the current dimension is smaller than the target, add padding
    #     pad_amount = target_shape[0] - current_shape[0]
    #     pad_start = pad_amount // 2
    #     pad_end = pad_amount - pad_start
    #     orig[pad_start:pad_end, :, :] = image
    #     label[pad_start:pad_end, :, :] = labels
    # else:
    #     # If the current dimension is larger than the target, crop the data
    #     crop_amount = current_shape[0] - target_shape[0]
    #     crop_start = crop_amount // 2
    #     crop_end = current_shape[0] - (crop_amount - crop_start)
    #     orig = image[crop_start:crop_end, :, :]
    #     label = labels[crop_start:crop_end, :, :]
    #
    # # Pad or crop on y-axis
    # if current_shape[1] < target_shape[1]:
    #     # If the current dimension is smaller than the target, add padding
    #     pad_amount = target_shape[1] - current_shape[1]
    #     pad_start = pad_amount // 2
    #     pad_end = pad_amount - pad_start
    #     orig[:, pad_start:pad_end, :] = image
    #     label[:, pad_start:pad_end, :] = labels
    # else:
    #     # If the current dimension is larger than the target, crop the data
    #     crop_amount = current_shape[1] - target_shape[1]
    #     crop_start = crop_amount // 2
    #     crop_end = current_shape[1] - (crop_amount - crop_start)
    #     orig = image[:, crop_start:crop_end, :]
    #     label = labels[:, crop_start:crop_end, :]
    #
    # # Pad or crop on z-axis
    # if current_shape[2] < target_shape[2]:
    #     # If the current dimension is smaller than the target, add padding
    #     pad_amount = target_shape[2] - current_shape[2]
    #     pad_start = pad_amount // 2
    #     pad_end = pad_amount - pad_start
    #     orig[:, :, pad_start:pad_end] = image
    #     label[:, :, pad_start:pad_end] = labels
    # else:
    #     # If the current dimension is larger than the target, crop the data
    #     crop_amount = current_shape[2] - target_shape[2]
    #     crop_start = crop_amount // 2
    #     crop_end = current_shape[2] - (crop_amount - crop_start)
    #     orig = image[:, :, crop_start:crop_end]
    #     label = labels[:, :, crop_start:crop_end]

    # for i in range(len(current_shape)):
    #     if current_shape[i] < target_shape[i]:
    #         # If the current dimension is smaller than the target, add padding
    #         pad_amount = target_shape[i] - current_shape[i]
    #         pad_start = pad_amount // 2
    #         pad_end = pad_amount - pad_start
    #         padding[i] = (pad_start, pad_end)
    #     elif current_shape[i] > target_shape[i]:
    #         # If the current dimension is larger than the target, crop the data
    #         crop_amount = current_shape[i] - target_shape[i]
    #         crop_start = crop_amount // 2
    #         crop_end = current_shape[i] - (crop_amount - crop_start)
    #         padding[i] = (crop_start, crop_end)

    # Pad or crop the original data
    # padded_image = np.pad(image, padding, mode='constant', constant_values=0)
    # padded_labels = np.pad(labels, padding, mode='constant', constant_values=0)
    # return padded_image, padded_labels
    return orig, label


def fix_orientation_and_voxel_size(img,
                                   labels,
                                   plane: str = 'coronal',
                                   out_shape: tuple = (256, 256, 256),
                                   vox_size: float = 1.0) -> tuple:
    """
    Aligns an MRI image and its corresponding labels with the RAS+ orientation by applying a 3D affine transformation.
    Depending on the specified 'plane', the function also permutes the axes to maintain consistent orientation across
    different imaging planes.

    Parameters
    ----------
    img : ndarray
        The original T1-weighted MRI volume.
    labels : ndarray
        The volume containing labeled regions of interest.
    plane : str, optional
        The anatomical plane (e.g., 'coronal', 'sagittal', 'axial') that influences the permutation of the axes.
        Default is 'coronal'.
    out_shape : tuple of int, optional
        The desired dimensions of the output volume (width, height, depth), defaulting to (256, 256, 256).
    vox_size : float, optional
        The isotropic spatial resolution in millimeters for each axis of the output volume, default is 1.0 mm.

    Returns
    -------
    tuple of ndarray
        The transformed MRI volume and label volume, both adjusted to the specified output shape and voxel size,
        aligned to the RAS+ space.
    """
    # Align the volumes to the RAS+ space and resample
    img, _ = reorient_resample_volume(img, vox_size, out_shape)
    labels, _ = reorient_resample_volume(labels, vox_size, out_shape, interpolation_order=0)

    # Change axis according to the desired plan
    img = apply_plane_orientation(img, plane)
    labels = apply_plane_orientation(labels, plane)

    # Ensure image shapes coincide
    assert img.shape == labels.shape, "Image and labels shapes must be equal."

    return img, labels


def reorient_resample_volume(volume: nib.Nifti1Image,
                             vox_zoom,
                             output_shape: tuple = (256, 256, 256),
                             interpolation_order: int = 1):
    """
        Reorients and resamples a 3D MRI volume to a specified voxel size and output shape.

        This function takes an MRI volume, reorients it to the standard anatomical orientation RAS+,
        and resamples it to a new voxel size and shape. The output volume has the dimensions specified
        in 'output_shape' and maintains the new voxel resolution defined by 'vox_zoom'.

        Parameters
        ----------
        volume : nib.Nifti1Image
            The input MRI volume in NIfTI format to be reoriented and resampled.
        vox_zoom : float
            The desired voxel size (in mm). The voxel size defines the spatial resolution of the resampled image on
            all three axes
        output_shape : tuple of int, optional
            The dimensions (width, height, depth) of the output volume in voxels, by default (256, 256, 256).
        interpolation_order : int, optional
            The order of the spline interpolation to be used for resampling. The default is 1, which corresponds to
            linear interpolation. Higher values provide more accurate interpolation at the cost of computational resources.

        Returns
        -------
        ndarray
            The reoriented and resampled MRI volume as an ndarray, with the specified voxel size and output dimensions.
    """
    # Change images orientation and resample
    # 1) Compute the center coordinates vector in the current XYZ space (in mm)
    # Get the midpoint in voxel coordinates and convert to homogeneous coordinates
    vol_center = np.hstack((np.array(volume.shape) / 2.0, [1]))
    # Transform the voxel midpoint to real-world coordinates using the volume's affine matrix
    center_xyz = np.dot(volume.affine, vol_center)

    # 2) Calculate the new volume center in the target isotropic space (256x256x256 mm)
    # Compute the center in real-world coordinates based on the desired output shape and voxel
    vol_center = [s / 2 * z for s, z in zip(output_shape, (vox_zoom, vox_zoom, vox_zoom))]

    # 3) Define the affine transform that maps new voxel space coordinates to real-world coordinates
    # Create an affine transformation matrix where each voxel corresponds to the specified voxel size
    # and the center of the volume is aligned with the new center coordinates
    a = np.array([[vox_zoom, 0, 0, center_xyz[0] - vol_center[0]],
                  [0, vox_zoom, 0, center_xyz[1] - vol_center[1]],
                  [0, 0, vox_zoom, center_xyz[2] - vol_center[2]],
                  [0, 0, 0, 1]])

    # 4) Compute the compound affine that maps current voxel coordinates to new voxel space
    # Note: Affine transformations are applied in reverse order when composing them
    b = inv(a) @ volume.affine

    # 5) Apply the compound affine matrix and resample the image
    # Use scipy's affine_transform to resample the image data according to the computed affine transformation
    img_data = volume.get_fdata()
    reoriented_volume = affine_transform(img_data, inv(b), output_shape=output_shape, order=interpolation_order)

    return reoriented_volume, a


def apply_plane_orientation(image, plane: str):
    if plane == 'axial':
        image = image.transpose((2, 1, 0))
    elif plane == 'coronal':
        image = image.transpose((1, 2, 0))
    elif plane == 'sagittal':
        image = image.transpose((0, 2, 1))
    return image


def fix_orientation_inference(img, plane):
    if plane == 'axial':
        img = img.transpose((2, 1, 0))
    elif plane == 'coronal':
        img = img.transpose((1, 2, 0))
    else:
        img = img.transpose((0, 2, 1))
    return img


def revert_fix_orientation_inference(img, plane):
    if plane == 'axial':
        img = img.permute((3, 1, 2, 0))
    elif plane == 'coronal':
        img = img.permute((3, 1, 0, 2))
    else:
        img = img.permute((0, 1, 3, 2))
    return img


def orient2coronal(img, plane):
    if plane == 'axial':
        img = img.transpose((2, 1, 0))
    elif plane == 'coronal':
        img = img.transpose((1, 2, 0))
    else:
        img = img.transpose((0, 2, 1))
    return img


def load_subjects(subjects: list,
                  plane: str,
                  data_padding: list,
                  slice_thickness: int,
                  lut: list,
                  lut_sag: list,
                  right_left_dict: dict,
                  preprocessing_mode: str,
                  loss_function: str,
                  mode: str = '',
                  save_hdf5: bool = False,
                  unilateral_classes: bool = False):
    # Lists for images, labels, label weights, and zooms
    images = []
    labels = []
    weights = []
    zooms = []  # (X, Y, Z) -> physical dimensions (in mm) of a voxel along each axis

    for subject in subjects:
        # Extract: orig (original images), orig_labels (annotations according to the
        # FreeSurfer convention), zooms (voxel dimensions)
        try:
            img = nib.load(os.path.join(subject, ORIG))
            img_labels = nib.load(os.path.join(subject, LABELS))
        except Exception as e:
            print(f'Exception loading: {subject}: {e}')
            continue

        # Transform according to the current plane.
        # Performed prior to removing blank slices.
        img_data, img_labels = fix_orientation_and_voxel_size(img,
                                                              img_labels,
                                                              plane)

        if preprocessing_mode is not None:
            # Initialize a transformations list
            transforms_list = []

            # 2) Normalize
            transforms_list.extend(get_norm_transforms(preprocessing_mode))

            # 3) Apply transformations:
            # Stack all images into a single 3D array
            stacked_images = np.expand_dims(img_data, axis=0)

            # Create a TorchIO ScalarImage instance
            image = tio.ScalarImage(tensor=stacked_images)

            # Apply each transformation
            for transform in transforms_list:
                image = transform(image)

            # Retrieve the transformed NumPy array
            transformed_image_array = image.data.numpy()

            # Split the transformed array back into individual slices
            img_data = np.squeeze(transformed_image_array)

        # Normalize to [0, 255]
        min_val = np.min(img_data)
        max_val = np.max(img_data)
        if min_val != max_val:
            img_data = (img_data - min_val) * (255 / (max_val - min_val))
        else:
            img_data = np.zeros_like(img_data)

        # Convert to uint8 and int16 respectively
        img_data = np.asarray(img_data, dtype=np.uint8)
        img_labels = np.asarray(img_labels, dtype=np.int16)

        # Add Gaussian Noise on training data
        # if mode == 'train':
        #     img_data = add_gaussian_noise(data=img_data,
        #                                   std_dev=5,
        #                                   mean=0)

        # Map the labels starting with 0
        new_labels = lut2labels(labels=img_labels,
                                lut_labels=lut,
                                right_left_map=right_left_dict,
                                plane=plane)

        # Compute class weights
        weights_array, weights_list = compute_weights(new_labels,
                                                      loss_function)
        if save_hdf5:
            # Save the fused labels and weights as well
            fused_labels = lut2labels(labels=img_labels,
                                      lut_labels=lut_sag,
                                      right_left_map=right_left_dict,
                                      plane=plane,
                                      unilat=unilateral_classes)
            fused_weights, fused_weights_list = compute_weights(fused_labels,
                                                                loss_function)
            return img_data, new_labels, weights_array, fused_labels, fused_weights, weights_list, fused_weights_list

        else:
            # Create thick slices
            create_thick_slices(slice_thickness,
                                img_data,
                                new_labels,
                                weights_array)

        # Append the new subject to the dataset
        images.extend(img_data)
        labels.extend(new_labels)
        weights.extend(weights_array)

    return images, labels, weights


def create_thick_slices(slice_thickness,
                        img_data: np.ndarray,
                        labels: np.ndarray,
                        weights: np.ndarray):
    # Create an MRI slice window => (D, slice_thickness, H, W)
    if slice_thickness > 1:
        img_data = get_thick_slices(img_data,
                                    slice_thickness)
        img_data = img_data.transpose((0, 3, 1, 2))
    else:
        img_data = np.expand_dims(img_data, axis=1)

    # Remove blank slices
    img_data, img_labels, weights_array = remove_blank_slices(images=img_data,
                                                              labels=labels,
                                                              weights=weights)
    # Return the results
    return img_data, img_labels, weights_array


def mni2coronal(img):
    """
    Transpose images from MNI space to coronal
    """
    return img.transpose((1, 2, 0))


def mni2axial(img):
    """
    Transpose images from MNI space to axial
    """
    return img.transpose((2, 1, 0))


def mni2sagittal(img):
    """
    Transpose images from MNI space to sagittal
    """
    return img.transpose((0, 2, 1))


def sagittal2full(pred: torch.Tensor,
                  lut: dict,
                  lut_sagittal: dict,
                  right_left_map: dict):
    """
    Transform sagittal predictions to full (78 classes)
    """
    full_class_list = []
    for i in lut.keys():
        if i in lut_sagittal.keys():
            full_class_list.append(lut_sagittal[i])
        else:
            if i >= 2000:
                full_class_list.append(lut_sagittal[i - 1000])
            else:
                full_class_list.append(right_left_map[lut[i]])
    return pred[:, full_class_list, :, :]


def add_gaussian_noise(data,
                       std_dev,
                       mean):
    """
    Adds Gaussian noise to data array
    """
    noise = np.random.normal(mean, std_dev, size=data.shape)

    # Add the noise to the MRI data
    data += noise

    # Clip the values to ensure they remain within the range of 0 to 255
    noisy_data = np.clip(data, 0, 255)

    return noisy_data


def preprocess_dataset(images: list,
                       padding: list = (320, 320, 320),
                       mode: str = 'percentiles_&_zscore'):
    """
    Performs cropping, normalization, followed by augmentation.
    1) Crop or pad:
        - Set a standard input size.
        - If data shape is above the standard size => crop
        - If data shape is below the standard size => pad
    2) Normalization:
    """
    # 1) Crop or pad
    # for i in range(len(images)):
    # images[i], labels[i] = crop_or_pad(images[i], labels[i], padding)
    #############

    # Initialize a transformations list
    transforms_list = []

    # 2) Normalize
    transforms_list.extend(get_norm_transforms())

    # 3) Apply transformations:
    # Stack all images into a single 3D array
    stacked_images = np.stack(images, axis=0)
    stacked_images = np.expand_dims(stacked_images, axis=0)

    # Create a TorchIO ScalarImage instance
    image = tio.ScalarImage(tensor=stacked_images)

    # Apply each transformation
    for transform in transforms_list:
        image = transform(image)

    # Retrieve the transformed NumPy array
    transformed_image_array = image.data.numpy()

    # Split the transformed array back into individual slices
    transformed_image_array = np.squeeze(transformed_image_array)
    transformed_images = [np.squeeze(image) for image in np.split(transformed_image_array, len(images), axis=0)]

    return transformed_images


def preprocess_subject(image: np.ndarray,
                       preprocessing_mode: str,
                       padding=(320, 320, 320)):
    """
    Performs cropping, normalization, followed by augmentation.
    1) Crop or pad:
        - Set a standard input size.
        - If data shape is above the standard size => crop
        - If data shape is below the standard size => pad
    2) Normalization:
    """
    # 1) Initialize a transformations list
    transforms_list = []

    # 2) Normalize
    transforms_list.extend(get_norm_transforms(preprocessing_mode))

    # 3) Apply transformations:
    # Create a TorchIO ScalarImage instance
    image = np.expand_dims(image, axis=0)
    image = tio.ScalarImage(tensor=image)

    # Apply each transformation
    for transform in transforms_list:
        image = transform(image)

    # Retrieve the transformed NumPy array
    transformed_image_array = image.data.numpy()

    return np.squeeze(transformed_image_array, axis=0)


def get_norm_transforms(mode: str = 'percentiles_&_zscore') -> (np.ndarray, np.ndarray):
    """
    Provides data normalization transforms.
    Normalization types:
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

    Parameters
    ----------
    mode
        'percentiles_&_zscore' or 'log_norm_&_zscore'
    """
    # Initialize a transforms list:
    transforms_list = []

    if mode == 'percentiles_&_zscore':
        # Append the RescaleIntensity and Z-score Normalization transforms
        transforms_list.append(tio.ZNormalization(masking_method=lambda x: x > 0))
        transforms_list.append(tio.RescaleIntensity(out_min_max=(0, 255), percentiles=(1, 99)))
    elif mode == 'zscore':
        # Append Z-score Normalization
        transforms_list.append(tio.ZNormalization(masking_method=lambda x: x > 0))
    else:
        return []
    return transforms_list


def get_aug_transforms(data_augmentation: str):
    """
    Provides data augmentation transforms
    See U-net paper:  https://arxiv.org/pdf/1809.10486.pdf (section Data Augmentation)
    See TorchIO -> Transforms -> Augmentation: https://torchio.readthedocs.io/transforms/augmentation.html
    """
    # Check if any augmentation method is needed
    if data_augmentation is None:
        return None

    # Get the augmentation list
    augs = data_augmentation.split(',')

    # Transforms list
    transform_list = []

    # Append each transform
    for aug in augs:
        if aug == 'Rotation':
            transform_list.append(
                tio.RandomAffine(
                    scales=(1.0, 1.0),
                    degrees=10,
                    translation=(0, 0, 0),
                    isotropic=True,
                    center='image',
                    default_pad_value='minimum',
                    image_interpolation='linear',
                    include=['img', 'label', 'weight'],
                ))
        elif aug == 'Scaling':
            transform_list.append(
                tio.RandomAffine(
                    scales=(0.8, 1.15),
                    degrees=0,
                    translation=(0, 0, 0),
                    isotropic=True,
                    center='image',
                    default_pad_value='minimum',
                    image_interpolation='linear',
                    include=['img', 'label', 'weight'],
                )
            )
        elif aug == 'Translation':
            transform_list.append(
                tio.RandomAffine(
                    scales=(1.0, 1.0),
                    degrees=0,
                    translation=(15.0, 15.0, 0),
                    isotropic=True,
                    center="image",
                    default_pad_value="minimum",
                    image_interpolation="linear",
                    include=["img", "label", "weight"]
                )
            )
        elif aug == 'Gamma':
            transform_list.append(
                tio.transforms.RandomGamma(
                    log_gamma=(-0.3, 0.3), include=['img']
                )
            )

    # Return
    return tio.Compose(transform_list)


def get_thick_slices(img_data,
                     slice_thickness: int = 3):
    """
    Creates a sliding window view into the volume with the given slice thickness.

    Parameters
    ----------
    img_data : np.ndarray
        3D array representing the volume of image data.

    slice_thickness : int, optional
        The thickness of each slice, by default 3. This determines how many slices
        are combined to form each output slice.

    Returns
    -------
    np.ndarray
        A new view into the original volume with a sliding window applied.
        Each window corresponds to a thick slice of the original data.
    """
    # Pad the image data on the first dimension (slice axis) with half the slice thickness on each side.
    # This ensures that the sliding window can produce thick slices that extend evenly around each original slice.
    img_data = np.pad(img_data, pad_width=((slice_thickness // 2, slice_thickness // 2), (0, 0), (0, 0)), mode="edge")

    # Create a sliding window view of the padded image data.
    # This view creates thick slices along the first dimension of the image data.
    return np.lib.stride_tricks.sliding_window_view(img_data, slice_thickness, axis=0)
# endregion


# region Data split
def get_train_test_split(subjects: list,
                         train_split: float = 0.8,
                         test_split: float = 0):
    """
    Creates a train/val/test split
    """
    # Shuffle the subjects
    random.shuffle(subjects)

    # Get the dataset length
    subjects_count = len(subjects)

    # Compute train split size and training set
    train_size = int(train_split * subjects_count)
    train_set = subjects[:train_size]

    test_set = []
    if test_split != 0:
        val_split = 1 - train_split - test_split
        val_size = round(val_split * subjects_count)
        test_set = subjects[train_size + val_size:]
    else:
        val_size = subjects_count - train_size

    # Compute the validation set
    val_set = subjects[train_size: train_size + val_size]

    return train_set, val_set, test_set


# endregion


# region Post-processing
def lateralize_volume(volume, subcort_right_left_map, unified_labels: True):
    """
    Lateralize the volume back to the original 95 class labels
    """
    if unified_labels:
        # All symmetric labels are unified
        # Identify the white matter mask
        wm = volume == 2

        # Define a structuring element and perform a binary opening
        # to remove connections
        structuring_element = np.ones((5, 5, 5), dtype=bool)
        wm = binary_opening(wm, structure=structuring_element)

        # Use scikit-image to compute the region props
        props = measure.regionprops(measure.label(wm, background=False, connectivity=2))
        sorted_props = sorted(props, key=lambda x: x.area, reverse=True)
        largest_centroids = [prop.centroid for prop in sorted_props[:2]]

        # Determine which centroid is on the right and which
        # one is on the left using the X coordinate
        if largest_centroids[0][0] > largest_centroids[1][0]:
            right_wm_centroid, left_wm_centroid = largest_centroids[0], largest_centroids[1]
        else:
            right_wm_centroid, left_wm_centroid = largest_centroids[1], largest_centroids[0]
    else:
        # Get left/right white matter centroids directly
        left_wm_centroid = get_class_centroid(volume == 2)
        right_wm_centroid = get_class_centroid(volume == 41)

    if unified_labels:
        # Remap the subcortical labels based on the distance towards the centroids
        for right, left in subcort_right_left_map.items():
            # Get left label mask
            label_mask = volume == left

            # Get the coordinates of the current label
            coords = np.argwhere(label_mask)

            # Calculate distances to centroids
            left_distances = np.linalg.norm(coords - left_wm_centroid, axis=1)
            right_distances = np.linalg.norm(coords - right_wm_centroid, axis=1)

            # Determine which coordinates are closer to the right centroid
            right_mask = right_distances < left_distances

            # Update the volume according to the right_mask
            right_coords = coords[right_mask]
            for coord in right_coords:
                volume[tuple(coord)] = right

    # List of all left-hand cortical classes that have been merged before training
    if unified_labels:
        cortical_classes = [1002, 1003, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017,
                            1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1034,
                            1035]
    else:
        cortical_classes = [1003, 1006, 1007, 1008, 1009, 1011, 1015, 1018, 1019, 1020, 1026, 1027,
                            1029, 1030, 1031, 1034, 1035]

    # Map each pixel to the nearest centroid for each label
    for label in cortical_classes:
        # Create a mask for the current label
        label_mask = volume == label

        # Get the coordinates of the current label
        coords = np.argwhere(label_mask)

        # Calculate distances to centroids
        left_distances = np.linalg.norm(coords - left_wm_centroid, axis=1)
        right_distances = np.linalg.norm(coords - right_wm_centroid, axis=1)

        # Determine which coordinates are closer to the right centroid
        right_mask = right_distances < left_distances

        # Update the volume: Add 1000 to the positions where right_distance is smaller
        # Use the original coordinates filtered by right_mask
        right_coords = coords[right_mask]
        for coord in right_coords:
            volume[tuple(coord)] += 1000

    return volume


def get_class_centroid(volume):
    """
    Returns the centroid of a class (computed only on the largest connected region)
    """
    # Calculate region properties
    props = measure.regionprops(measure.label(volume, background=False, connectivity=3))
    little_props = measure.label(volume, background=False)

    # Find the largest region by area
    largest_region_index = np.argmax([prop.area for prop in props])

    # Compute the centroid of the largest region
    centroid = props[largest_region_index].centroid

    return centroid
# endregion
