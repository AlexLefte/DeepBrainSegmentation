import numpy
import h5py
import argparse
import os
import json

import numpy as np

import data_utils as du

from sklearn.model_selection import KFold
import random

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dataset Creation")

    # Define the argument for the data path
    parser.add_argument('--data_path',
                        type=str,
                        default='/dataset',
                        help='Path towards subjects directory.')

    # Define the argument for the output path of the dataset.
    parser.add_argument('--output',
                        type=str,
                        default='/dataset/',
                        help='Output path towards the training datasets.')

    # Define the argument for the dataset name
    parser.add_argument('--dataset_name',
                        type=str,
                        default='dataset_revised_test',
                        help='Output path towards the training datasets.')

    # Define the argument for the validation split percentage.
    parser.add_argument('--val_split',
                        type=float,
                        default=0.1,
                        help='Train/test sets for cross validation.')

    # Define the argument to specify whether to create cross-validation folds.
    parser.add_argument('--create_folds',
                        type=bool,
                        default=True,
                        help='Create training folds.')

    # Parse the command-line arguments and store them in 'args'
    args = parser.parse_args()

    # Get the current directory.
    current_dir = os.path.dirname(__file__)

    # Construct the path to the parent directory.
    parent_dir = os.path.dirname(os.path.dirname(current_dir))

    # Read the JSON configuration file located in the parent directory.
    cfg = json.load(open(parent_dir + '/config/config.json', 'r'))

    # Get all paths to the subject directories in the data path and shuffle them
    data_path = parent_dir + args.data_path
    subject_paths = [os.path.join(data_path, s) for s in os.listdir(data_path)
                     if os.path.isdir(os.path.join(data_path, s))]
    random.shuffle(subject_paths)

    # Initialize a list to store subject splits for training and validation
    subject_splits = []
    if args.create_folds:
        # Create n separate folds and save them inside a dataset
        n_splits = int(np.floor(1 / args.val_split))
        kf = KFold(n_splits=n_splits, shuffle=True)
        splits = kf.split(subject_paths)

        # Determine the training and validation subjects for each split
        for i, split in enumerate(splits):
            train_subjects = [os.path.basename(subject_paths[i]) for i in split[0]]
            val_subjects = [os.path.basename(subject_paths[i]) for i in split[1]]

            # Append the tuple of train and val subjects to the subject_splits list
            subject_splits.append((train_subjects, val_subjects))

    # Obtain the list of data augmentation transforms from the configuration.
    transform = du.get_aug_transforms(cfg['data_augmentation'])

    # Get data processing modality from the configuration.
    processing_modality = cfg['preprocessing_modality']

    # Obtain the data padding settings from the configuration.
    data_padding = [int(x) for x in cfg['data_padding'].split(',')]

    # Get the slice thickness for the images from the configuration.
    slice_thickness = cfg['slice_thickness']

    # Get unilateral flag
    unilateral_classes = cfg['unilateral_classes']

    # Obtain the color look-up table and the left-right dictionary from the configuration.
    lut = du.get_lut(cfg['base_path'] + cfg['lut_path'])
    lut_labels = lut["ID"].values
    lut_labels_sagittal = du.get_sagittal_labels_from_lut(lut)
    right_left_dict = du.get_right_left_dict(lut)

    # Define the path for the HDF5 dataset.
    dataset_path = parent_dir + f'/dataset/{args.dataset_name}.hdf5'

    # Create an HDF5 file to save the dataset.
    with h5py.File(dataset_path, "w") as hf:
        # Create a dataset to save the list of subjects.
        hf.create_dataset('subjects', data=subject_paths, dtype=h5py.special_dtype(vlen=str))

        # Save cross-validation folds if required
        if args.create_folds:
            splits_group = hf.create_group('splits')
            for i, split in enumerate(subject_splits):
                split_group = splits_group.create_group(f'split_{i}')
                split_group.create_dataset('train', data=split[0], dtype=h5py.special_dtype(vlen=str))
                split_group.create_dataset('val', data=split[1], dtype=h5py.special_dtype(vlen=str))

        # Save slices, labels and weights for each subject, each orientation (axial, coronal, sagittal)
        for subject in subject_paths:
            # For each subject, load and save the data.
            images, labels, weights, fused_labels, fused_weights, weights_list, fused_weights_list = du.load_subjects(
                subjects=[subject],
                plane='',
                data_padding=data_padding,
                slice_thickness=slice_thickness,
                lut=lut_labels,
                lut_sag=lut_labels_sagittal,
                right_left_dict=right_left_dict,
                preprocessing_mode=processing_modality,
                loss_function=cfg[
                    'loss_function'],
                save_hdf5=True,
                unilateral_classes=unilateral_classes)

            # Convert data to specific formats for efficient storage.
            images = np.asarray(images, dtype=np.uint8)
            labels = np.asarray(labels, dtype=np.uint8)
            weights = np.asarray(weights, dtype=float)
            fused_labels = np.asarray(fused_labels, dtype=np.uint8)
            fused_weights = np.asarray(fused_weights, dtype=float)
            weights_list = np.asarray(weights_list, dtype=float)
            fused_weights_list = np.asarray(fused_weights_list, dtype=float)

            # Save the subject under the respective plane group within the split
            subject_group = hf.create_group(os.path.basename(subject))
            subject_group.create_dataset("images", data=images)
            subject_group.create_dataset("labels", data=labels)
            subject_group.create_dataset("weights", data=weights)
            subject_group.create_dataset("fused_labels", data=fused_labels)
            subject_group.create_dataset("fused_weights", data=fused_weights)
            subject_group.create_dataset("weights_list", data=weights_list)
            subject_group.create_dataset("fused_weights_list", data=fused_weights_list)

    # Print a success message at the end.
    print(f'Dataset {args.dataset_name} was successfully saved.')
