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


ORIG = 't1weighted_brain.MNI152.nii.gz'
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
                 weights_dict: list = None):
        """
        Constructor
        """
        # Get the subjects' names
        # self.subjects = [s for s in os.listdir(path) if os.path.isdir(os.path.join(path, s))]
        self.subjects = subjects

        # Save the mode:
        self.mode = mode

        # Get the appropriate transformation list
        self.transform = du.get_aug_transforms() if mode == 'train' else None

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
        self.weights_dict = {} if self.mode == 'train' else weights_dict
        self.weights = [] if self.mode == 'train' else None

        # Get the color look-up tables and right-left dictionary
        self.lut = du.get_lut(cfg['base_path'] + cfg['lut_path'])
        self.lut_labels = self.lut["ID"].values if self.plane != 'sagittal' \
            else du.get_sagittal_labels_from_lut(self.lut)
        self.right_left_dict = du.get_right_left_dict(self.lut)

        # Get start time and load the data
        start_time = time.time()
        for subject in self.subjects:
            # Extract: orig (original images), orig_labels (annotations according to the
            # FreeSurfer convention), zooms (voxel dimensions)
            try:
                img = nib.load(os.path.join(subject, ORIG))
                img_data = img.get_fdata()
                zooms = img.header.get_zooms()
                img_labels = np.asarray(nib.load(os.path.join(subject, LABELS)).get_fdata())
            except Exception as e:
                print(f'Exception loading: {subject}: {e}')
                continue

            # Transform according to the current plane.
            # Performed prior to removing blank slices.
            img_data, zooms, img_labels = du.fix_orientation(img_data,
                                                             zooms,
                                                             img_labels,
                                                             self.plane)

            if img_data.shape != (218, 182, 182):
                print(img_data.shape)

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

        # # Plot some slices before processing:
        # indexes = range(110, 150, 10)
        # slice_list = [self.images[i] for i in range(len(self.images)) if i in indexes]
        # du.plot_slices(slice_list, 'Before processing')

        # Preprocess the data (based on statistics of the entire dataset)
        # self.images, self.labels = du.preprocess(self.images,
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
            image = np.expand_dims(image.T, axis=(0, 3))
            labels = np.expand_dims(labels.T, axis=(0, 3))
            weights = np.expand_dims(weights.T, axis=(0, 3))

            if image.shape != labels.shape:
                print("GetItem: Image and labels shapes must be equal.")

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
            weights = torch.squeeze(transform_result['weight'].data, dim=(0, -1)).t()
        else:
            image, labels, weights = self.images[idx], self.labels[idx], torch.ones(self.images[idx].shape)
            image = torch.Tensor(image).unsqueeze(dim=0)
            labels = torch.Tensor(labels)

        return {
            'image': image,
            'labels': labels,
            'weights': weights,
            'weights_dict': torch.tensor(list(self.weights_dict.values()))
        }
