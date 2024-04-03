import argparse
import time
from datetime import datetime
import os
import numpy as np
import torch
import logging
from data import data_utils as du
from data.data_loader import get_inference_data_loader
import json
from tqdm import tqdm
from pandas import DataFrame
from utils import logger
from src.models.fcnn_model import FCnnModel
from src.utils.nifti import save_nifti
import nibabel as nib
from src.utils.metrics import *

LOGGER = logging.getLogger(__name__)


def check_device(device: str):
    """
    Check device availability
    """
    if device == 'cuda':
        if torch.cuda.is_available():
            LOGGER.info('Running inference on cuda...')
            return 'cuda'
        else:
            LOGGER.info('Cuda is not available. Running inference on cpu...')
            return 'cpu'
    else:
        LOGGER.info('Running inference on cpu...')
        return 'cpu'


def get_subjects(input_path):
    # Create the subjects list
    if os.path.isdir(input_path):
        subjects_list = [os.path.join(input_path, s) for s in os.listdir(input_path)]
    elif os.path.isfile(input_path):
        subjects_list = [input_path]
    else:
        raise ValueError(f"{input_path} is neither a directory nor a file.")
    return subjects_list


def get_subjects_and_labels(input_path, labels_path):
    # Create the subjects list
    if os.path.isdir(input_path):
        subjects_paths = sorted(os.listdir(input_path))
        labels_paths = sorted(os.listdir(labels_path))
        subjects_list = [os.path.join(input_path, s) for s in subjects_paths]
        labels_list = [os.path.join(labels_path, s) for s in labels_paths]
    elif os.path.isfile(input_path):
        subjects_list = [input_path]
        labels_list = [labels_path]
    else:
        raise ValueError(f"{input_path} is neither a directory nor a file.")
    return subjects_list, labels_list


def lateralize_volume(volume):
    """
    Lateralizes the volume
    """
    # TODO: To be implemented
    pass


def load_checkpoints(device: str,
                     params: dict):
    """
    Loads checkpoints
    """
    # Flush denormal numbers to zero to improve performance
    torch.set_flush_denormal(True)

    # Base checkpoint path
    checkpoints_path = "checkpoints/"

    # Initialize model dictionary
    models_dict = {
        'axial': None,
        'coronal': None,
        'sagittal': None
    }
    # models_dict = {
    #     'coronal': None
    # }

    # Save in_channels
    in_channels = params["in_channels"]

    # Load each model
    for plane in models_dict.keys():
        params["in_channels"] = in_channels
        params['num_classes'] = 51 if plane == 'sagittal' else 79
        ckp_path = os.path.join(checkpoints_path, plane, 'best.pkl')
        checkpoint = torch.load(ckp_path, map_location=device)
        model = FCnnModel(params).to(device)
        model.load_state_dict(checkpoint['model_state'])
        models_dict[plane] = model

    # Return the models dictionary
    return models_dict


def run_inference(models: dict,
                  input_path: str,
                  output_path: str,
                  cfg: dict,
                  device: str = 'cpu'):
    """
    Runs inference on all three FCNNs and aggregates the decision by weighted sum
    """
    # Load LUTs
    lut = du.get_lut(args.lut)
    # Get the LUT into the 0-78 range
    lut_labels_dict = {value: index for index, value in enumerate(lut['ID'])}
    # Get the sagittal LUT into the 0-51 range
    lut_labels_sag_dict = {value: index for index, value in enumerate(du.get_sagittal_labels_from_lut(lut))}
    # Create a map between right-left subcortical structures
    right_left_labels_map = du.get_right_left_dict(lut)
    # Create a map between the initial right-left subcortical structures
    right_left_lut_map = {lut_labels_dict[right]: lut_labels_dict[left] for right, left in right_left_labels_map.items()}

    # Get the subject list
    subjects_list = get_subjects(input_path)

    # Initialize the plane weights
    plane_weight = {
        'axial': 0.4,
        'coronal': 0.4,
        'sagittal': 0.2
    }

    # Initialize a prediction list
    predictions = {}
    # For testing purposes
    predictions_before_aggregation = {}

    # Get predictions for each subject
    for subject in subjects_list:
        # Log some info
        file_name = os.path.basename(subject)
        LOGGER.info(f'Running inference on {file_name}')

        aggregated_pred = None

        for plane, model in models.items():
            # Create an inference loader
            loader = get_inference_data_loader(subject, cfg, plane)

            # Prediction list
            pred_list = []

            # Start the timer
            start = time.time()
            for batch_idx, batch in tqdm(enumerate(loader)):
                # Get the slices, labels and weights, then send them to the desired device
                image = batch.to(device).float()

                # Initialize the prediction tensor
                if aggregated_pred is None:
                    d, h, w = loader.dataset.initial_shape
                    shape = (d, 79, h, w)
                    aggregated_pred = np.zeros(shape)

                # Set up the model to eval mode
                model.eval()

                with torch.inference_mode():
                    # Get the predictions
                    model_pred = model(image)

                    # Process sagittal predictions (since labels are lateralized)
                    if plane == 'sagittal':
                        model_pred = du.sagittal2full(pred=model_pred,
                                                      lut=lut_labels_dict,
                                                      lut_sagittal=lut_labels_sag_dict,
                                                      right_left_map=right_left_lut_map)
                    pred_list.append(model_pred.cpu().numpy())

            # Reorient the volume
            prediction_volume = np.concatenate(pred_list, axis=0)
            prediction_volume = du.revert_fix_orientation_inference(prediction_volume, plane)

            # Save the result before aggregation
            predictions_before_aggregation[plane] = prediction_volume.argmax(axis=1).astype(np.int16)

            # Aggregate the result
            aggregated_pred += plane_weight[plane] * prediction_volume
            end = time.time()
            LOGGER.info(f"==== Finished inference for plane {plane}. Total time: {end - start:.3f} seconds ===="
                        f"\n===========================================")
            print(f"==== Finished inference for plane {plane}. Total time: {end - start:.3f} seconds ===="
                  f"\n===========================================")

        # Apply argmax
        pred_classes = np.argmax(aggregated_pred, axis=1).astype(np.int16)

        # Map back to the initial LUT (FreeSurfer)
        # pred_classes = du.labels2lut(pred_classes, lut_labels_dict)

        # Relateralize the volume
        # pred_classes = du.lateralize_volume(pred_classes)

        # Append the prediction
        predictions[subject] = pred_classes

    # Save the predictions
    save_predictions(predictions,
                     predictions_before_aggregation,
                     output_path,
                     lut['ID'].values)

    return predictions, predictions_before_aggregation


def run_test(models: dict,
             labels_path: str,
             cfg: dict,
             device: str = 'cpu'):
    """
    Runs inference and performance metrics.
    Segmentation file is required.
    """
    start_time = time.time()
    preds, plane_preds = run_inference(models=models,
                                       input_path=args.input_path,
                                       output_path=args.output_path,
                                       cfg=cfg,
                                       device=args.device)

    # Get the flat predictions
    flat_preds = {}
    for sub, pred in preds.items():
        flat_preds[sub] = pred.flatten()
    for plane, pred in plane_preds.items():
        flat_preds[plane] = pred.flatten()

    # Load the ground truth
    labels = np.asarray(nib.load(labels_path).get_fdata(), dtype=np.int16).flatten()
    # cortical_classes = [1003, 1006, 1007, 1008, 1009, 1011, 1015, 1018, 1019, 1020, 1021, 1025, 1026, 1027,
    #                     1029, 1030, 1031, 1034, 1035]
    # cortical_classes.extend(du.get_lut(args.lut)['ID'])
    # # Process the labels: unknown => background
    # mask = ~np.isin(labels, cortical_classes)
    # # Use the mask to replace elements with 0
    # labels[mask] = 0

    # # TODO: Just for the moment we will get the labels into the 0-78 range. Remove afterwards.
    lut = du.get_lut(args.lut)
    right_left_dict = du.get_right_left_dict(lut)
    labels = du.lut2labels(labels=labels,
                           lut_labels=lut["ID"].values,
                           right_left_map=right_left_dict,
                           plane='coronal')
    labels = labels.flatten()

    # Compute the dsc for each of them:
    dsc = {}
    for key, flat_pred in flat_preds.items():
        dsc_tuple = get_cortical_subcortical_class_dsc(y_pred=flat_pred,
                                                       y_true=labels,
                                                       num_classes=79)
                                                       # classes=cortical_classes)
        dsc[key] = dsc_tuple

    # Print some results
    for key, dice_scores in dsc.items():
        scores = f'{dice_scores[0]} subcortical, {dice_scores[1]} cortical, {dice_scores[2]} mean.'
        print(f'Scores for {key}: {scores}')

    end_time = time.time()
    LOGGER.info(f"==== Stopped testing. Total time: {end_time - start_time:.3f} seconds ===="
                f"\n===========================================")


def save_predictions(predictions: dict,
                     predictions_without_aggregation: dict,
                     output_path: str,
                     lut: list):
    for subject, prediction in predictions.items():
        prediction = du.get_lut_from_labels(prediction,
                                            lut)
        subject_name = os.path.basename(subject)
        prediction_path = os.path.join(output_path, 'aggregated.nii')
        save_nifti(prediction, prediction_path)

    for plane, prediction in predictions_without_aggregation.items():
        prediction = du.get_lut_from_labels(prediction,
                                            lut)
        subject_name = os.path.basename(plane)
        prediction_path = os.path.join(output_path, f'{subject_name}_{plane}.nii')
        save_nifti(prediction, prediction_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Inference settings
    parser.add_argument('--input_path',
                        type=str,
                        default='dataset/OASIS-TRT-20-2/t1weighted.MNI152.nii.gz',
                        help='Path towards the input file/directory')

    parser.add_argument('--labels_path',
                        type=str,
                        default='dataset/OASIS-TRT-20-2/labels.DKT31.manual+aseg.MNI152.nii.gz',
                        help='Path towards the labels file/directory')

    parser.add_argument('--output_path',
                        type=str,
                        default='output/',
                        help='Path to the output directory')

    parser.add_argument('--device',
                        type=str,
                        default='cuda',
                        help='Device to perform inference on.')

    parser.add_argument('--cfg',
                        type=str,
                        default='config/config.json',
                        help='Path to config file')

    parser.add_argument('--lut',
                        type=str,
                        default='config/FastSurfer_ColorLUT.tsv')

    args = parser.parse_args()

    # If default output path was chosen, create a new directory
    if args.output_path == '../output/':
        args.output_path += datetime.now().strftime("%m-%d-%y_%H-%M")
        try:
            os.makedirs(args.output_path)
            print("Directory created successfully:", args.output_path)
        except Exception as e:
            print("Error occurred while creating the output directory:", e)

    # Setting up the device
    args.device = check_device(args.device)

    # Get the base_path
    base_path = os.getcwd()

    # Load the config file
    cfg = json.load(open(args.cfg, 'r'))

    # Load the best checkpoints
    models = load_checkpoints(args.device, cfg)

    # Check if testing is required
    if args.labels_path is not None:
        # Run testing
        run_test(models=models,
                 labels_path=args.labels_path,
                 cfg=cfg,
                 device=args.device)
    else:
        # Run inference for each input subject
        run_inference(models=models,
                      input_path=args.input_path,
                      output_path=args.output_path,
                      cfg=cfg,
                      device=args.device)






