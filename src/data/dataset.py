import logging
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchio
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

        # Save the mode:
        self.mode = mode

        # Get plane
        self.plane = cfg['plane']

        # Get image processing modality:
        self.processing_modality = cfg['preprocessing_modality']

        # Get data padding:
        self.data_padding = [int(x) for x in cfg['data_padding'].split(',')]

        # Lists for images, labels, label weights, and zooms
        self.images = []
        self.labels = []
        self.zooms = []  # (X, Y, Z) -> physical dimensions (in mm) of a voxel along each axis

        # Weights dictionary
        self.weights_dict = {} if self.mode == 'train' else None
        self.weights = [] if self.mode == 'train' else None

        # Get the color look-up tables and right-left dictionary
        self.lut = du.get_lut(cfg['lut_path'])
        self.lut_labels = self.lut["ID"].values if self.plane != 'sagittal' \
            else du.get_sagittal_labels_from_lut(self.lut)
        self.right_left_dict = du.get_right_left_dict(self.lut)

        # Get start time and load the data
        start_time = time.time()
        self.subjects = [self.subjects[1]]
        for subject in self.subjects:
            # Get subject path
            subject_path = os.path.join(self.data_path, subject)

            # Extract: orig (original images), orig_labels (annotations according to the
            # FreeSurfer convention), zooms (voxel dimensions)
            img = nib.load(os.path.join(subject_path, ORIG))
            img_data = img.get_fdata()
            zooms = img.header.get_zooms()
            img_labels = np.asarray(nib.load(os.path.join(subject_path, LABELS)).get_fdata())

            # Transform according to the current plane.
            # Performed prior to removing blank slices.
            img_data, zooms, new_labels = du.fix_orientation(img_data,
                                                             zooms,
                                                             img_labels,
                                                             self.plane)

            # Remove blank slices
            img_data, img_labels = du.remove_blank_slices(images=img_data,
                                                          labels=img_labels)

            # Map the labels starting with 0
            new_labels = du.get_labels(labels=img_labels,
                                       lut_labels=self.lut_labels,
                                       right_left_map=self.right_left_dict,
                                       plane=self.plane)

            # Append the new subject to the dataset
            self.images.extend(img_data)
            self.labels.extend(new_labels)
            self.zooms.extend((zooms, ) * img_data.shape[0])

        # Check the intensity statistics across the dataset (before preprocessing)
        # du.compare_intensity_across_subjects(self.images,
        #                                     self.subjects)
        # print("Statistics before preprocessing: ")
        # Stack the slices along a new axis (axis=0)
        # stacked_slices = np.stack(self.images, axis=0)
        # du.compare_intensity_across_dataset(stacked_slices)
        # du.plot_histogram(data=stacked_slices,
        #                   title='Histogram before preprocessing')

        if self.mode == 'train':
            # Compute class weigths
            self.weights, self.weights_dict = du.compute_weights(self.labels)
        unique_classes, counts = np.unique(self.weights, return_counts=True)

        # # Plot some slices before processing:
        # indexes = range(110, 150, 10)
        # slice_list = [self.images[i] for i in range(len(self.images)) if i in indexes]
        # du.plot_slices(slice_list, 'Before processing')

        # Preprocess the data (based on statistics of the entire dataset)
        self.images = du.preprocess(self.images,
                                    self.data_padding)

        # Check the intensity statistics across the dataset (after preprocessing)
        # print("\nStatistics after preprocessing: ")
        # stacked_slices = np.stack(self.images, axis=0)
        # du.compare_intensity_across_dataset(stacked_slices)
        # du.plot_histogram(data=stacked_slices,
        #                   title='Histogram after preprocessing')

        # # Plot some slices after processing:
        # slice_list = [self.images[i] for i in range(len(self.images)) if i in indexes]
        # du.plot_slices(slice_list, 'After processing')

        # Get the length of our Dataset
        self.count = len(self.images)

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
        image, labels, weights = self.images[idx], self.labels[idx], self.weights[idx]

        unique_classes, counts = np.unique(labels, return_counts=True)

        # Apply transforms if they exist
        if self.transform:
            image = np.expand_dims(image.T, axis=(0, 3))
            labels = np.expand_dims(labels.T, axis=(0, 3))
            weights = np.expand_dims(weights.T, axis=(0, 3))

            # image = np.expand_dims(image, axis=0)
            # image = np.expand_dims(image.transpose((2, 0, 1)), axis=3)
            # labels = labels[np.newaxis, :, :, np.newaxis]
            # weights = weights[np.newaxis, :, :, np.newaxis]

            # Create the subject dictionary
            subject_dict = {
                'img': torchio.ScalarImage(tensor=image),
                'label': torchio.LabelMap(tensor=labels),
                'weight': torchio.LabelMap(tensor=weights)
            }

            # Initialize a Subject instance
            subject = torchio.Subject(subject_dict)

            # Get the transformation results
            transform_result = self.transform(subject)
            image = torch.squeeze(transform_result['img'].data, dim=-1).permute(0, 2, 1)
            labels = torch.squeeze(transform_result['label'].data, dim=(0, -1)).t()
            unique_classes, counts = np.unique(labels, return_counts=True)
            weights = torch.squeeze(transform_result['weight'].data, dim=(0, -1)).t()
        else:
            image = torch.Tensor(image)
            labels = torch.Tensor(labels)
            weights = torch.Tensor(weights)

        unique_classes, counts = np.unique(weights, return_counts=True)

        # plt.figure()
        # plt.imshow(image, cmap='gray', origin='lower')
        # plt.title('After')
        # plt.show()

        return {
            'image': image,
            'labels': labels,
            'weights': weights,
            'weights_dict': torch.tensor(list(self.weights_dict.values()))
        }


class SubjectsDatasetTest(Dataset):
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

        # Save the mode:
        self.mode = mode

        # Get plane
        self.plane = cfg['plane']

        # Get image processing modality:
        self.processing_modality = cfg['preprocessing_modality']

        # Get data padding:
        self.data_padding = [int(x) for x in cfg['data_padding'].split(',')]

        # Lists for images, labels, label weights, and zooms
        self.images = []
        self.labels = []
        self.zooms = []  # (X, Y, Z) -> physical dimensions (in mm) of a voxel along each axis

        # Get the color look-up tables and right-left dictionary
        self.lut = du.get_lut(cfg['lut_path'])
        self.lut_labels = self.lut["ID"].values if self.plane != 'sagittal' \
            else du.get_sagittal_labels_from_lut(self.lut)
        self.right_left_dict = du.get_right_left_dict(self.lut)

        # Get start time and load the data
        start_time = time.time()
        self.subjects = [self.subjects[1]]
        for subject in self.subjects:
            # Get subject path
            subject_path = os.path.join(self.data_path, subject)

            # Extract: orig (original images), orig_labels (annotations according to the
            # FreeSurfer convention), zooms (voxel dimensions)
            img = nib.load(os.path.join(subject_path, ORIG))
            img_data = img.get_fdata()
            zooms = img.header.get_zooms()
            img_labels = np.asarray(nib.load(os.path.join(subject_path, LABELS)).get_fdata())

            # Transform according to the current plane.
            # Performed prior to removing blank slices.
            img_data, zooms, new_labels = du.fix_orientation(img_data,
                                                             zooms,
                                                             img_labels,
                                                             self.plane)

            # Remove blank slices
            img_data, img_labels = du.remove_blank_slices(images=img_data,
                                                          labels=img_labels)

            # Map the labels starting with 0
            new_labels = du.get_labels(labels=img_labels,
                                       lut_labels=self.lut_labels,
                                       right_left_map=self.right_left_dict,
                                       plane=self.plane)

            # Append the new subject to the dataset
            self.images.extend(img_data)
            self.labels.extend(new_labels)
            self.zooms.extend((zooms, ) * img_data.shape[0])

        # Preprocess the data (based on statistics of the entire dataset)
        self.images = du.preprocess(self.images,
                                    self.data_padding,
                                    self.mode)

        # Get the lenght of our Dataset
        self.count = len(self.images)

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

        # Apply transforms if they exist
        if self.transform:
            # subject = torchio.Subject(image=torchio.ScalarImage(image),
            #                           labels=torchio.LabelMap(labels))
            subject = torchio.Subject(
                {
                    'img': torchio.ScalarImage(tensor=image),
                    'label': torchio.LabelMap(tensor=labels)
                }
            )
            transform_result = self.transform(subject)
            image = torch.squeeze(transform_result['img'].data, dim=-1)
            labels = torch.squeeze(transform_result['label'].data, dim=-1)
        else:
            image = torch.Tensor(image)
            labels = torch.Tensor(labels)

        return {
            'image': image,
            'labels': labels,
        }
