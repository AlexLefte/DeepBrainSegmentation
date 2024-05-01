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

    parser.add_argument('--data_path',
                        type=str,
                        default='/dataset',
                        help='Path towards subjects directory.')

    parser.add_argument('--output',
                        type=str,
                        default='/dataset/',
                        help='Output path towards the training datasets.')

    parser.add_argument('--dataset_name',
                        type=str,
                        default='dataset_splits_test',
                        help='Output path towards the training datasets.')

    parser.add_argument('--val_split',
                        type=float,
                        default=0.1,
                        help='Train/test sets for cross validation.')

    parser.add_argument('--create_folds',
                        type=bool,
                        default=True,
                        help='Create training folds.')

    args = parser.parse_args()

    # Load the configuration file
    # Get the current directory of the script
    current_dir = os.path.dirname(__file__)

    # Construct the path to the parent directory
    parent_dir = os.path.dirname(os.path.dirname(current_dir))

    # Read the configuration file
    cfg = json.load(open(parent_dir + '/config/config.json', 'r'))

    # Get the subjects' paths
    data_path = parent_dir + args.data_path
    subject_paths = [os.path.join(data_path, s) for s in os.listdir(data_path)
                     if os.path.isdir(os.path.join(data_path, s))]
    random.shuffle(subject_paths)

    # Create training/validation splits
    subject_splits = []
    if args.create_folds:
        # Create n separate folds and save them inside a dataset
        n_splits = int(np.floor(1 / args.val_split))
        kf = KFold(n_splits=n_splits, shuffle=True)
        splits = kf.split(subject_paths)
        for i, split in enumerate(splits):
            train_subjects = [os.path.basename(subject_paths[i]) for i in split[0]]
            val_subjects = [os.path.basename(subject_paths[i]) for i in split[1]]

            # Append the tuple of train and val subjects to subject_splits
            subject_splits.append((train_subjects, val_subjects))

    # Get the appropriate transformation list
    transform = du.get_aug_transforms(cfg['data_augmentation'])

    # Get image processing modality:
    processing_modality = cfg['preprocessing_modality']

    # Get data padding:
    data_padding = [int(x) for x in cfg['data_padding'].split(',')]

    # Get slice thickness
    slice_thickness = cfg['slice_thickness']

    # # Split the train/test subject sets
    # train, val, test = du.get_train_test_split(subject_paths,
    #                                            cfg['train_size'],
    #                                            cfg['test_size'])

    # # Splits
    # splits = {
    #     'train': train,
    #     'val': val
    # }

    # Define the planes list
    planes = ['axial', 'coronal', 'sagittal']

    # Get the color look-up tables and right-left dictionary
    lut = du.get_lut(cfg['base_path'] + cfg['lut_path'])
    lut_labels = lut["ID"].values
    lut_labels_sagittal = du.get_sagittal_labels_from_lut(lut)
    right_left_dict = du.get_right_left_dict(lut)

    # Define the dataset path
    dataset_path = parent_dir + f'/dataset/{args.dataset_name}.hdf5'

    # Create a hdf5 dataset for each split
    with h5py.File(dataset_path, "w") as hf:
        # Save subjects list
        hf.create_dataset('subjects', data=subject_paths, dtype=h5py.special_dtype(vlen=str))

        # Save cross-validation folds if required
        if args.create_folds:
            splits_group = hf.create_group('splits')
            for i, split in enumerate(subject_splits):
                split_group = splits_group.create_group(f'split_{i}')
                split_group.create_dataset('train', data=split[0], dtype=h5py.special_dtype(vlen=str))
                split_group.create_dataset('val', data=split[1], dtype=h5py.special_dtype(vlen=str))

        # Save slices, labels and weights for each subject, each orientation (axial, coronal, sagittal)
        for plane in planes:
            plane_group = hf.create_group(plane)
            for subject in subject_paths:
                # Load and save the subjects
                images, labels, weights = du.load_subjects(subjects=[subject],
                                                           plane=plane,
                                                           data_padding=data_padding,
                                                           slice_thickness=slice_thickness,
                                                           lut=lut_labels if plane != 'sagittal' else lut_labels_sagittal,
                                                           right_left_dict=right_left_dict,
                                                           preprocessing_mode=processing_modality,
                                                           loss_function=cfg['loss_function'])

                # Convert to uint8
                images = np.asarray(images, dtype=np.uint8)
                labels = np.asarray(labels, dtype=np.uint8)
                weights = np.asarray(weights, dtype=float)

                # Save the subject under the respective plane group within the split
                subject_group = plane_group.create_group(os.path.basename(subject))
                subject_group.create_dataset("images", data=images)
                subject_group.create_dataset("labels", data=labels)
                subject_group.create_dataset("weights", data=weights)
                # plane_group.create_dataset("zooms", data=zooms)  # Unused for the moment

    # Print success message
    print(f'Dataset {args.dataset_name} was successfully saved.')
