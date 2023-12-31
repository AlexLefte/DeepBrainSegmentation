import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchio as tio
import nibabel.orientations as orientations
from torchvision import transforms
import logging


LOGGER = logging.getLogger(__name__)


def compute_weights(labels: list) -> (list, dict):
    """
    Computes the classes weights matrix

    Parameters
    ----------
    labels: np.ndarray
        image labels
    """
    # Initialize the weights list
    weights_list = []

    # Stacks the labelled slices:
    stacked_labels = np.stack(labels, axis=0)

    # Get the unique values in the label matrix
    unique_classes, count = np.unique(stacked_labels, return_counts=True)

    # Compute the median
    median_count = np.median(count)

    # Convert to float type
    count = np.array(count, dtype=float)

    # Compute the weight for each class and save it into the dictionary
    for i in range(len(count)):
        count[i] = float(median_count) / count[i]

    # Create the class weights dictionary
    weights_dict = dict(zip(unique_classes, count))

    # Define a weight ndarray for each training slice
    for label_array in labels:
        # Apply the mapping to the original matrix
        weights_array = np.vectorize(weights_dict.get)(label_array)

        # Append the new weight matrix to the collection:
        weights_list.append(weights_array)
    return weights_list, weights_dict


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


# def get_labels_from_lut(path: str) -> dict.keys:
#     """
#     Get labels from LUT
#     """
#     return get_lut(path).keys()


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
#####################


# Data Preprocessing
def remove_blank_slices(images: np.ndarray,
                        labels: np.ndarray,
                        threshold: int = 20):
    """
    Removes slices with very few labeled voxels.

    Parameters
    ----------
    images:
        the MRI volume as a
    labels:
        the labeled volume
    threshold:
        the minimum sum a slice has to accomplish in order to be kept

    Returns
    -------
    The volumes without those slices that do not meet the requirements
    """
    # Compute the sums of the labels for each slice
    slices_sums = np.sum(labels, axis=(1, 2))

    # Select those slices with at least 20 voxels different from background
    selected_slices = np.where(slices_sums > threshold)

    # # Plot some blank slices:
    # unselected_slices = np.where(slices_sums <= threshold)
    # for i in unselected_slices[0][0::5]:
    #     plt.figure(), plt.imshow(images[i, :, :], cmap='gray')
    #     plt.show()

    # Return the selected slices
    return images[selected_slices], labels[selected_slices]


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
    padding = [(0, 0), (0, 0)]

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
    padded_image = np.pad(image, padding, mode='constant', constant_values=0)
    padded_labels = np.pad(labels, padding, mode='constant', constant_values=0)
    return padded_image, padded_labels


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

    # if plane == 'axial':
    #     img = np.moveaxis(img, [0, 1, 2], [1, 0, 2])
    #     labels = np.moveaxis(labels, [0, 1, 2], [1, 0, 2])
    #     zooms = zooms[1:]
    # elif plane == 'sagittal':
    #     img = np.moveaxis(img, [0, 1, 2], [2, 1, 0])
    #     labels = np.moveaxis(labels, [0, 1, 2], [2, 1, 0])
    #     zooms = zooms[:2]
    # else:
    #     zooms = zooms[:2]

    if plane == 'axial':
        img = img.transpose((1, 0, 2))
        labels = labels.transpose((1, 0, 2))
        zooms = zooms[1:]
    elif plane == 'sagittal':
        img = img.transpose((2, 1, 0))
        labels = labels.transpose((2, 1, 0))
        zooms = zooms[:2]
    else:
        zooms = zooms[:2]
    return img, zooms, labels


def preprocess(images: list,
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
    #     images[i], labels[i] = crop_or_pad(images[i], labels[i], padding)
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

    # Split the transformed array back into individual images
    transformed_image_array = np.squeeze(transformed_image_array)
    transformed_images = [np.squeeze(image) for image in np.split(transformed_image_array, len(images), axis=0)]

    return transformed_images


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
        # Append the RescaleIntensity and ZNormalization transforms
        transforms_list.append(tio.RescaleIntensity(percentiles=(0.5, 99.5)))
        transforms_list.append(tio.ZNormalization())
    # elif mode == 'log_norm_&_zscore':
    #     # Append the ZNormalization transform
    #     transforms_list.append(tio.ZNormalization())
    #     # Apply log transform on the image
    #     # Note: TorchIO does not implement such a scaling mode
    #     stacked_images = np.log(stacked_images + 1)
    else:
        # Wrong mode => return initial image
        # LOGGER.info("Invalid normalization mode.")
        return []
    return transforms_list


def get_aug_transforms():
    """
    Provides data augmentation transforms
    See U-net paper:  https://arxiv.org/pdf/1809.10486.pdf (section Data Augmentation)
    See TorchIO -> Transforms -> Augmentation: https://torchio.readthedocs.io/transforms/augmentation.html
    """

    # Append: rotation -> scaling -> translation -> elastic deformation -> gamma correction
    return tio.Compose([
        tio.RandomAffine(
            scales=(1.0, 1.0),
            degrees=10,
            translation=(0, 0, 0),
            isotropic=True,
            center='image',
            default_pad_value='minimum',
            image_interpolation='linear',
            include=['img', 'label', 'weight'],
        ),
        tio.RandomAffine(
            scales=(0.8, 1.15),
            degrees=0,
            translation=(0, 0, 0),
            isotropic=True,
            center='image',
            default_pad_value='minimum',
            image_interpolation='linear',
            include=['img', 'label', 'weight'],
        ),
        tio.RandomAffine(
            scales=(1.0, 1.0),
            degrees=0,
            translation=(15.0, 15.0, 0),
            isotropic=True,
            center="image",
            default_pad_value="minimum",
            image_interpolation="linear",
            include=["img", "label", "weight"]
        ),
        # tio.RandomElasticDeformation(
        #     num_control_points=7,
        #     max_displacement=15,
        #     locked_borders=4,
        #     image_interpolation='linear',
        #     include=['img', 'label', 'weight'],
        # ),
        tio.transforms.RandomGamma(
            log_gamma=(-0.3, 0.3), include=['img']
        )]
    )
####################


