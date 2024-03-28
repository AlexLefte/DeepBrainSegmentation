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
import h5py
from src.utils.nifti import save_nifti


ORIG = 't1weighted.MNI152.nii.gz'
LABELS = 'labels.DKT31.manual+aseg.MNI152.nii.gz'
LOGGER = logging.getLogger(__name__)


class SubjectsDataset(Dataset):
    """
    Class used to load the MRI scans into a custom dataset
    """
    def __init__(self,
                 cfg: dict,
                 subjects: list,
                 mode: str,
                 weights_dict: dict = None):
        """
        Constructor
        """
        # Get the subjects' names
        self.subjects = subjects

        # Save the mode:
        self.mode = mode

        # Get the appropriate transformation list
        self.transform = du.get_aug_transforms(cfg['data_augmentation']) if mode == 'train' else None

        # Get plane
        self.plane = cfg['plane']

        # Get image processing modality:
        self.processing_modality = cfg['preprocessing_modality']

        # Get data padding:
        self.data_padding = [int(x) for x in cfg['data_padding'].split(',')]

        # Get slice thickness
        self.slice_thickness = cfg['slice_thickness']

        # Get the color look-up tables and right-left dictionary
        self.lut = du.get_lut(cfg['base_path'] + cfg['lut_path'])
        self.lut_labels = self.lut["ID"].values if self.plane != 'sagittal' \
            else du.get_sagittal_labels_from_lut(self.lut)
        self.right_left_dict = du.get_right_left_dict(self.lut)

        # Get start time and load the subjects
        start_time = time.time()
        if cfg['hdf5_dataset']:
            hdf5_file = os.path.join(cfg['base_path'], cfg['data_path'], cfg['hdf5_file'])
            # Load the images and labels from the HDF5 file
            with h5py.File(hdf5_file, "r") as hf:
                mode_group = hf[mode]
                plane_group = mode_group[self.plane]
                self.images = plane_group['images'][:]
                self.labels = plane_group['labels'][:]
                # self.zooms = plane_group['zooms'][:]
        else:
            # Load the subjects directly
            self.images, self.labels, self.zooms = du.load_subjects(self.subjects,
                                                                    self.plane,
                                                                    self.data_padding,
                                                                    self.slice_thickness,
                                                                    self.lut_labels,
                                                                    self.right_left_dict,
                                                                    self.processing_modality)

        if self.mode == 'train':
            # Get the loss function type
            loss_fn = cfg['loss_function']

            # Compute class weights
            self.weights, self.weights_dict = du.compute_weights(self.labels,
                                                                 loss_fn)
        elif self.mode == 'val':
            self.weights_dict = weights_dict
            self.weights = du.get_weights_list(self.labels,
                                               self.weights_dict)
        else:
            self.weights = []
            self.weights_dict = {}

        # Get the length of our Dataset
        self.count = len(self.images)

        # Get stop time and display info
        stop_time = time.time()
        LOGGER.info(f'{self.mode} dataset loaded in {stop_time - start_time: .3f} s.\n'
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
        # Apply transforms if they exist
        if self.transform is not None:
            image, labels, weights = self.images[idx], self.labels[idx], self.weights[idx]
            image = np.expand_dims(image, axis=3)
            labels = np.expand_dims(labels, axis=(0, 3))
            weights = np.expand_dims(weights, axis=(0, 3))

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
            image = torch.squeeze(transform_result['img'].data, dim=-1)
            labels = torch.squeeze(transform_result['label'].data, dim=(0, -1))
            weights = torch.squeeze(transform_result['weight'].data, dim=(0, -1))
        else:
            if self.mode == 'train' or self.mode == 'val':
                image, labels, weights = self.images[idx], self.labels[idx], self.weights[idx]
            else:
                image, labels, weights = self.images[idx], self.labels[idx], torch.ones_like(self.labels[idx])
            image = torch.Tensor(image)
            labels = torch.Tensor(labels)

        # Normalize the slice's values
        image_min = image.min()
        image_max = image.max()
        image = (image - image_min) / (image_max - image_min)

        return {
            'image': image,
            'labels': labels,
            'weights': weights,
            'weights_list': torch.tensor(list(self.weights_dict.values()))
        }


class InferenceSubjectsDataset(Dataset):
    """
    Class used to load only the MRI scans (excluding segmentation labels) into a custom dataset
    """

    def __init__(self,
                 cfg: dict,
                 subjects: list,
                 plane: str):
        """
        Constructor
        """
        # Get the subjects' names
        self.subjects = subjects

        # Get image processing modality:
        self.processing_modality = cfg['preprocessing_modality']

        # Get data padding:
        self.data_padding = [int(x) for x in cfg['data_padding'].split(',')]

        # Get slice thickness
        self.slice_thickness = cfg['slice_thickness']

        # Lists for images and zooms
        self.images = []
        self.zooms = []  # (X, Y, Z) -> physical dimensions (in mm) of a voxel along each axis

        # Get start time and load the data
        start_time = time.time()
        for subject in self.subjects:
            # Extract: orig (original images), zooms (voxel dimensions)
            try:
                img = nib.load(subject)
                img_data = img.get_fdata()
                zooms = img.header.get_zooms()
            except Exception as e:
                print(f'Exception loading: {subject}: {e}')
                continue

            # Save the initial shape of the volume
            self.initial_shape = img_data.shape

            # Transform according to the current plane.
            # Performed prior to removing blank slices.
            img_data = du.fix_orientation_inference(img_data, plane)

            # Preprocess the data (based on statistics of the entire dataset)
            img_data = du.preprocess_subject(img_data,
                                             self.processing_modality,
                                             self.data_padding)

            # Normalize the images to [0.0, 255.0]
            min_val = np.min(img_data)
            max_val = np.max(img_data)
            img_data = (img_data - min_val) * (255 / (max_val - min_val))
            img_data = np.asarray(img_data, dtype=np.uint8)

            # Create an MRI slice window => (D, slice_thickness, H, W)
            if self.slice_thickness > 1:
                img_data = du.get_thick_slices(img_data,
                                               self.slice_thickness)
                img_data = img_data.transpose((0, 3, 1, 2))
            else:
                img_data = np.expand_dims(img_data, axis=1)

            # Append the new subject to the dataset
            self.images.extend(img_data)
            self.zooms.extend((zooms,) * img_data.shape[0])

        # Get the length of our Dataset
        self.count = len(self.images)

        # Get stop time and display info
        stop_time = time.time()
        LOGGER.info(f'Inference dataset loaded in {stop_time - start_time: .3f} s.\n'
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
        # Normalize the slice's values
        image = self.images[idx]

        # Normalize the slice's values
        image_min = image.min()
        image_max = image.max()
        image = (image - image_min) / (image_max - image_min)

        return image

