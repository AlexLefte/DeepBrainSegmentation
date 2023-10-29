import logging
import time

import numpy as np
import torch
from torch.utils.data import Dataset
import os
import nibabel as nib
from src.data import data_utils as du
from torchvision import transforms


ORIG = 't1weighted.nii.gz'
LABELS = 'labels.DKT31.manual+aseg.nii.gz'
LOGGER = logging.getLogger(__name__)


class SubjectsDataset(Dataset):
    """
    Class used to load the MRI scans into a custom dataset
    """

    def __init__(self,
                 cfg: dict,
                 path: str,
                 mode: str,
                 transform: transforms = None,
                 target_transform: transforms = None):
        """
        Constructor
        """
        self.data_path = path
        self.transform = transform
        self.target_transform = target_transform

        # Get the subjects' names
        self.subjects = [s for s in os.listdir(path) if os.path.isdir(os.path.join(path, s))]
        self.count = len(self.subjects)

        # Save the mode:
        self.mode = mode

        # Get plane
        self.plane = cfg['plane']

        # Lists for images, labels, label weights, and zooms
        self.images = []
        self.labels = []
        self.zooms = []  # (X, Y, Z) -> physical dimensions (in mm) of a voxel along each axis

        # Weights dictionary
        self.weights = {} if self.mode == 'train' else None

        # Get the color look-up tables and right-left dictionary
        self.lut = du.get_lut(cfg['lut_path'])
        self.lut_labels = self.lut["ID"].values if self.plane != 'sagittal' \
            else du.get_sagittal_labels_from_lut(self.lut)
        self.right_left_dict = du.get_right_left_dict(self.lut)

        # Get start time and load the data
        start_time = time.time()
        for subject in self.subjects:
            # Get subject path
            subject_path = os.path.join(self.data_path, subject)

            # Extract: orig (original images), orig_labels (annotations according to the
            # FreeSurfer convention), zooms (voxel dimensions)
            img = nib.load(os.path.join(subject_path, ORIG))
            img_data = img.get_fdata()
            zooms = img.header.get_zooms()
            img_labels = np.asarray(nib.load(os.path.join(subject_path, LABELS)).get_fdata())

            # Map the labels starting with 0
            new_labels = du.get_labels(labels=img_labels,
                                       lut_labels=self.lut_labels,
                                       right_left_map=self.right_left_dict,
                                       plane=self.plane)

            # Append the new subject to the dataset
            self.images.append(img_data)
            self.labels.append(new_labels)
            self.zooms.append(zooms)

        if mode == 'train':
            # Compute the weights (useful in the median frequency balanced logistic loss)
            for label in self.labels:
                # Get the unique values in the label matrix
                unique_classes, count = np.unique(label, return_counts=True)

                for uc in unique_classes:
                    if uc in self.weights.keys():
                        self.weights[uc] += count
                    else:
                        self.weights[uc] = count

            # Compute the median
            median_count = np.median(list(self.weights.values()))

            for weight, class_count in self.weights.items():
                self.weights[weight] = median_count / class_count

        # Check the intensity statistics across the dataset
        du.compare_intensity_across_dataset(self.images,
                                            self.subjects)

        # Preprocessing the data
        self.__preprocess()

        # Get stop time and display info
        stop_time = time.time()
        LOGGER.info(f'{self.mode} dataset loaded in {stop_time - stop_time: .3f} s.\n'
                    f'Dataset length: {self.count}.')

    def __len__(self):
        """
        Returns the length of the custom dataset.
        Must be implemented.
        """
        return self.count

    def __getitem__(self, idx):
        """
        Returns the image data of the patient and the labels.
        Must be implemented.
        """
        image, labels = self.images[idx], self.labels[idx]

        # # TODO: transformare log
        # # Normalize the image in the [0, 1] range
        # for slice in np.shape(image, 0):
        #     image[slice, :, :] = (image[slice, :, :] - np.amin(image[slice, :, :])) / (np.amax(image[slice]) - np.amin(image[slice]))

        # Apply transforms if they exist
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            labels = self.target_transform(labels)

        return {
            "image": image,
            "labels": labels,
        }

    def __crop_or_pad(self, slice):
        pass

    def __preprocess(self):
        pass