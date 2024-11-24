import logging
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchio
from torch.utils.data import Dataset
import os
import nibabel as nib
import nibabel.processing
from src.data import data_utils as du
import h5py
from src.utils.nifti import save_nifti

ORIG = 't1weighted.nii.gz'
LABELS = 'labels.DKT31.manual+aseg.nii.gz'
LOGGER = logging.getLogger(__name__)


class SubjectsDataset(Dataset):
    """
    Class used to load the MRI scans into a custom dataset
    """

    def __init__(self,
                 cfg: dict,
                 subjects: list,
                 mode: str):
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

        # Get class fuse flag
        self.unilateral_classes = cfg['unilateral_classes']

        # Get class number
        self.num_classes = cfg['num_classes']

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
                self.images = []
                self.labels = []
                self.weights = []
                self.weights_list = []
                for subject_name, subject in hf.items():
                    if os.path.basename(subject_name) in self.subjects:
                        images = subject['images'][:]
                        if self.plane == 'sagittal' or self.unilateral_classes:
                            labels = subject['fused_labels'][:]
                            weights = subject['fused_weights'][:]
                            weights_list = [subject['fused_weights_list'][:]] * weights.shape[0]
                        else:
                            labels = subject['labels'][:]
                            weights = subject['weights'][:]
                            weights_list = [subject['weights_list'][:]] * weights.shape[0]

                        # Get the plane orientation
                        images = du.apply_plane_orientation(images, self.plane)
                        labels = du.apply_plane_orientation(labels, self.plane)
                        weights = du.apply_plane_orientation(weights, self.plane)
                        # [images, labels, weights] = du.apply_plane_orientation([images, labels, weights], self.plane)

                        # Create thick slice
                        images, labels, weights = du.create_thick_slices(self.slice_thickness,
                                                                         images,
                                                                         labels,
                                                                         weights)

                        # Append the results
                        self.images.extend(images)
                        self.labels.extend(labels)
                        self.weights.extend(weights)
                        self.weights_list.extend(weights_list)
        else:
            # Load the subjects directly
            self.images, self.labels, self.weights = du.load_subjects(self.subjects,
                                                                      self.plane,
                                                                      self.data_padding,
                                                                      self.slice_thickness,
                                                                      self.lut_labels,
                                                                      [],
                                                                      self.right_left_dict,
                                                                      self.processing_modality,
                                                                      cfg['loss_function'],
                                                                      mode,
                                                                      self.unilateral_classes)

        # if self.mode == 'test':
        #     self.weights = []
        #     self.weights_dict = {}
        # self.weights_dict = {}

        # Save the number of classes
        self.num_classes = cfg['num_classes']

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
            image, labels, weights, weights_list = (self.images[idx],
                                                    self.labels[idx],
                                                    self.weights[idx],
                                                    self.weights_list[idx])
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
                image, labels, weights, weights_list = (self.images[idx],
                                                        self.labels[idx],
                                                        self.weights[idx],
                                                        self.weights_list[idx])
            else:
                image, labels, weights, weights_list = (self.images[idx],
                                                        self.labels[idx],
                                                        torch.tensor(np.ones_like(self.labels[idx])),
                                                        self.weights_list[idx])
            image = torch.Tensor(image)
            labels = torch.Tensor(labels)

        # Divide by 255 and clip between 0.0 and 1.0
        image = image.float()
        image = torch.clamp(image / 255.0, min=0.0, max=1.0)

        return {
            'image': image,
            'labels': labels,
            'weights': weights,
            'weights_list': weights_list
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
                zooms = img.header.get_zooms()
            except Exception as e:
                print(f'Exception loading: {subject}: {e}')
                continue

            # Compute the output shape
            # self.initial_shape = (cfg['out_shape'],) * 3
            # vox_sizes = cfg['vox_size']
            # output_shape = tuple(int(a * b / vox_sizes) for a, b in zip(self.initial_shape, zooms))
            output_shape = (cfg['out_shape'],) * 3
            img_data, affine = du.reorient_resample_volume(img, vox_zoom=1.0, output_shape=output_shape)
            self.affine = affine

            # Save the initial output shape of the volume
            # self.initial_shape = output_shape

            # Initialize a transformations list
            transforms_list = []

            # 2) Normalize
            transforms_list.extend(du.get_norm_transforms('percentiles_&_zscore'))

            # 3) Apply transformations:
            # Stack all images into a single 3D array
            stacked_images = np.expand_dims(img_data, axis=0)

            # Create a TorchIO ScalarImage instance
            import torchio as tio
            image = tio.ScalarImage(tensor=stacked_images)

            # Apply each transformation
            for transform in transforms_list:
                image = transform(image)

            # Retrieve the transformed NumPy array
            transformed_image_array = image.data.numpy()

            # Split the transformed array back into individual slices
            img_data = np.squeeze(transformed_image_array)

            # Normalize the images to [0.0, 255.0]
            min_val = np.min(img_data)
            max_val = np.max(img_data)
            if min_val != max_val:
                img_data = (img_data - min_val) * (255 / (max_val - min_val))
            else:
                img_data = np.zeros_like(img_data)
            img_data = np.asarray(img_data, dtype=np.uint8)

            # Save the reoriented MRI
            save_nifti(img_data, os.path.join(cfg['output_path'], 'standard_image.nii'))

            # Change axis according to the desired plan
            if plane == 'axial':
                img_data = img_data.transpose((2, 1, 0))
            elif plane == 'coronal':
                img_data = img_data.transpose((1, 2, 0))
            else:
                img_data = img_data.transpose((0, 2, 1))

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
        image = np.clip(image / 255.0, a_min=0.0, a_max=1.0)

        return image
