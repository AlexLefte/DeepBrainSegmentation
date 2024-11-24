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
ORIG = 't1weighted.nii.gz'
LABELS = 'labels.DKT31.manual+aseg.nii.gz'


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
        subjects_list = [os.path.join(input_path, s, ORIG) for s in os.listdir(input_path)]
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
    num_classes = params['num_classes']

    # Load each model
    for plane in models_dict.keys():
        params["in_channels"] = in_channels
        params['num_classes'] = 51 if plane == 'sagittal' else num_classes
        ckp_path = os.path.join(checkpoints_path, plane, 'best.pkl')
        checkpoint = torch.load(ckp_path, map_location=device)
        model = FCnnModel(params).to(device)
        model.load_state_dict(checkpoint['model_state'])
        models_dict[plane] = model

        # Make sure the initial number of classes is being used
        params['num_classes'] = num_classes

    # Return the models dictionary
    return models_dict


def run_inference(models: dict,
                  input_path: str,
                  output_path: str,
                  cfg: dict,
                  device: str = 'cpu',
                  return_prediction: bool = True):
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
    right_left_lut_map = {lut_labels_dict[right]: lut_labels_dict[left] for right, left in
                          right_left_labels_map.items()}

    # Get the subject list
    subjects_list = get_subjects(input_path)

    # Unilateral?
    unilateral_classes = cfg['unilateral_classes']

    # Initialize the plane weights
    if unilateral_classes:
        plane_weight = {
            'axial': 1 / 3,
            'coronal': 1 / 3,
            'sagittal': 1 / 3
        }
    else:
        plane_weight = {
            'axial': 0.4,
            'coronal': 0.4,
            'sagittal': 0.2
        }

    # Initialize a prediction list
    predictions = {}
    # For testing purposes
    predictions_before_aggregation = {}

    times = []
    # Get predictions for each subject
    for subject in subjects_list:
        start_time = time.time()
        # Log some info
        file_name = os.path.join(subject)
        # print(f'Running inference on {file_name}')
        LOGGER.info(f'Running inference on {file_name}')

        aggregated_pred = None
        affine = None

        for plane, model in models.items():
            # Create an inference loader
            loader = get_inference_data_loader(subject, cfg, plane)
            affine = loader.dataset.affine

            # Prediction list
            pred_list = []

            # Batch index
            batch_index = 0

            # Start the timer
            start = time.time()
            for batch_idx, batch in tqdm(enumerate(loader)):
                # Get the slices, labels and weights, then send them to the desired device
                image = batch.to(device).float()

                # Initialize the prediction tensor
                if aggregated_pred is None:
                    d = cfg['out_shape']
                    shape = (d, cfg['num_classes'], d, d)
                    aggregated_pred = torch.zeros(shape, device='cpu')

                # Set up the model to eval mode
                model.eval()

                with (torch.inference_mode()):
                    # Get the predictions
                    model_pred = model(image)

                    # Process sagittal predictions (since labels are lateralized)
                    if (plane == 'sagittal') and (unilateral_classes is False):
                        model_pred = du.sagittal2full(pred=model_pred,
                                                      lut=lut_labels_dict,
                                                      lut_sagittal=lut_labels_sag_dict,
                                                      right_left_map=right_left_lut_map)
                    # pred_list.append(model_pred)

                    # Revert back to the original orientation
                    model_pred = du.revert_fix_orientation_inference(model_pred, plane)

                    # Add predictions for this batch
                    if plane == 'axial':
                        aggregated_pred[:, :, :, batch_index:batch_index + model_pred.shape[3]] += \
                            plane_weight[plane] * model_pred.cpu()
                        index = 3
                    elif plane == 'coronal':
                        aggregated_pred[:, :, batch_index:batch_index + model_pred.shape[2], :] += \
                            plane_weight[plane] * model_pred.cpu()
                        index = 2
                    else:
                        aggregated_pred[batch_index:batch_index + model_pred.shape[0], :, :, :] += \
                            plane_weight[plane] * model_pred.cpu()
                        index = 0
                    batch_index += model_pred.shape[index]

            # Reorient the volume
            # prediction_volume = torch.cat(pred_list, dim=0)
            # prediction_volume = du.revert_fix_orientation_inference(prediction_volume, plane)
            #
            # # Save the result before aggregation
            # # predictions_before_aggregation[plane] = prediction_volume.argmax(axis=1).astype(np.int16)
            #
            # # Aggregate the result
            # aggregated_pred += plane_weight[plane] * prediction_volume.cpu()
            end = time.time()
            LOGGER.info(f"==== Finished inference for plane {plane}. Total time: {end - start:.3f} seconds ===="
                        f"\n===========================================")
            # print(f"==== Finished inference for plane {plane}. Total time: {end - start:.3f} seconds ===="
            #       f"\n===========================================")

        # Apply argmax
        # pred_classes = np.argmax(aggregated_pred, axis=1).astype(np.uint8)
        pred_classes = aggregated_pred.argmax(dim=1).cpu().numpy().astype(np.uint8)

        # Map back to the initial LUT (FreeSurfer)
        pred_classes = du.labels2lut(pred_classes, lut_labels_sag_dict if unilateral_classes else lut_labels_dict)

        # Relateralize the volume
        pred_classes = du.lateralize_volume(pred_classes, right_left_labels_map, unilateral_classes)

        end_time = time.time()
        times.append(end_time - start_time)

        # Append the prediction
        predictions[subject] = pred_classes

    # Save the predictions
    save_predictions(predictions,
                     predictions_before_aggregation,
                     output_path,
                     lut['ID'].values,
                     affine)

    # return predictions, predictions_before_aggregation
    print(f'Timp mediu de inferenta: {np.mean(times)} secunde')
    LOGGER.info(f'Timp mediu de inferenta: {np.mean(times)} secunde')

    if return_prediction:
        return predictions, predictions_before_aggregation, subjects_list
    else:
        return


def run_test(models: dict,
             labels_path: str,
             cfg: dict,
             device: str = 'cpu'):
    """
    Runs inference and performance metrics.
    Segmentation file is required.
    """
    start_time = time.time()
    preds, plane_preds, subjects_list = run_inference(models=models,
                                                      input_path=args.input_path,
                                                      output_path=args.output_path,
                                                      cfg=cfg,
                                                      device=args.device)

    # Get the flat predictions
    flat_preds = {}
    for sub, pred in preds.items():
        sub = os.path.basename(os.path.normpath(os.path.dirname(sub)))
        flat_preds[sub] = pred.flatten()
    # for plane, pred in plane_preds.items():
    #     flat_preds[plane] = pred.flatten()

    # Load ground truth
    flat_labels = {}
    class_list = [2003, 2006, 2007, 2008, 2009, 2011, 2015, 2018, 2019, 2020, 2026, 2027,
                  2029, 2030, 2031, 2034, 2035]
    class_list.extend(du.get_lut(args.lut)['ID'])
    class_list = sorted(class_list)
    for subject in subjects_list:
        subject = os.path.dirname(subject)
        # Load the ground truth
        labels = nib.load(os.path.join(subject, LABELS))
        # labels = np.asarray(labels.get_fdata(), dtype=np.int16).flatten()

        output_shape = (cfg['out_shape'],) * 3
        labels, affine = du.reorient_resample_volume(labels,
                                                     vox_zoom=1.0,
                                                     output_shape=output_shape,
                                                     interpolation_order=0)
        # save_nifti(labels, f'/home/alex/PycharmProjects/DeepBrainSegmentation/output/labels_conformed.nii', affine=affine)
        labels = np.asarray(labels, dtype=np.int16).flatten()

        subject_name = os.path.basename(os.path.normpath(subject))

        # Process the labels: unknown => background
        mask = ~np.isin(labels, class_list)
        # Use the mask to replace elements with 0
        labels[mask] = 0

        flat_labels[subject_name] = labels

    # Create text file
    # Get the current date
    current_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Define the filename with the current date
    filename = f'test_{current_date}.txt'

    # Compute the dsc for the aggregated model per subject
    scores = {}
    for sub, flat_pred in flat_preds.items():
        # dsc_tuple = get_cortical_subcortical_class_dsc(y_pred=flat_pred,
        #                                                y_true=flat_labels[sub],
        #                                                classes=cortical_classes)
        scores_dict = get_scores(y_pred=flat_pred,
                                 y_true=flat_labels[sub],
                                 class_list=class_list[1:])

        scores[sub] = scores_dict
    with open(filename, 'a') as file:
        for key, s in scores.items():
            file.write(f'Subject: {key}\n')
            for k, v in s.items():
                file.write(f'Score {k}: {v}\n')
        file.write('\n')

    # Compute overall scores
    # Extract the numpy arrays from the dictionary
    arrays_list = list(flat_preds.values())
    # Concatenate the arrays
    flat_preds_all = np.concatenate(arrays_list)

    arrays_list = list(flat_labels.values())
    flat_labels_all = np.concatenate(arrays_list)

    scores_dict = get_scores(y_pred=flat_preds_all,
                             y_true=flat_labels_all,
                             class_list=class_list[1:])
    with open(filename, 'a') as file:
        for k, v in scores_dict.items():
            file.write(f'Score {k}: {v}\n')

    dice_scores = get_class_dsc(flat_preds_all,
                                flat_labels_all,
                                class_list=class_list,
                                return_mean=False)
    with open(filename, 'a') as file:
        for i, class_i in enumerate(class_list):
            file.write(f"Class {class_i} dice score: {dice_scores[i]}\n")

    end_time = time.time()
    LOGGER.info(f"==== Stopped testing. Total time: {end_time - start_time:.3f} seconds ===="
                f"\n===========================================")


def save_predictions(predictions: dict,
                     predictions_without_aggregation: dict,
                     output_path: str,
                     lut: list,
                     affine=None):
    for subject, prediction in predictions.items():
        # prediction = du.get_lut_from_labels(prediction,
        #                                     lut)
        subject_name = os.path.dirname(os.path.basename(subject))
        prediction_path = os.path.join(output_path, f'{subject_name}_aggregated.nii')
        save_nifti(prediction, prediction_path)

    for plane, prediction in predictions_without_aggregation.items():
        prediction = du.get_lut_from_labels(prediction,
                                            lut)
        subject_name = os.path.basename(plane)
        prediction_path = os.path.join(output_path, f'{subject_name}_{plane}.nii')
        save_nifti(prediction, prediction_path)


def main(input_path, labels_path, output_path, device, config_path, lut_path):
    """
    Main function
    """
    # Setare device și verificare disponibilitate
    device = check_device(device)

    # Încărcare configurație
    cfg = json.load(open(config_path, 'r'))
    cfg['output_path'] = output_path

    # Setare logger
    LOG_PATH = os.path.join(output_path, 'log_file.log')
    logger.create_logger(LOG_PATH)
    LOGGER = logging.getLogger(__name__)

    # Încărcare modele
    models = load_checkpoints(device, cfg)

    # Executie inferență sau testare, în funcție de prezența etichetelor
    if labels_path:
        run_test(models=models, labels_path=labels_path, cfg=cfg, device=device)
    else:
        run_inference(models=models, input_path=input_path, output_path=output_path, cfg=cfg, device=device,
                      return_prediction=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Inference settings
    parser.add_argument('--input_path',
                        type=str,
                        default='dataset/OASIS-TRT-20-8/t1weighted.nii.gz',
                        # default=None,
                        help='Path towards the input file/directory')

    parser.add_argument('--labels_path',
                        type=str,
                        # default='dataset/NKI-TRT-20-20/labels.DKT31.manual+aseg.nii.gz',
                        default=None,
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
    # if args.output_path == '../output/':
    #     args.output_path += datetime.now().strftime("%m-%d-%y_%H-%M")
    #     try:
    #         os.makedirs(args.output_path)
    #         # print("Output directory created successfully:", args.output_path)
    #         LOGGER.info("Output directory created successfully:", args.output_path)
    #     except Exception as e:
    #         # print("Error occurred while creating the output directory:", e)
    #         LOGGER.error("Error occurred while creating the output directory:", e)

    main(args.input_path, args.labels_path, args.output_path, args.device, args.cfg, args.lut)

    # # Setting up the device
    # args.device = check_device(args.device)
    #
    # # Get the base_path
    # base_path = os.getcwd()
    #
    # # Load the config file
    # cfg = json.load(open(args.cfg, 'r'))
    #
    # # Add output path to the configuration dictionary
    # cfg['output_path'] = args.output_path
    #
    # # Setup the logger
    # LOG_PATH = os.path.join(base_path, 'log_file' + '.log')
    # logger.create_logger(LOG_PATH)
    # LOGGER = logging.getLogger(__name__)
    #
    # # Load the best checkpoints
    # models = load_checkpoints(args.device, cfg)
    #
    # # Check if testing is required
    # if args.labels_path is not None:
    #     # Run testing
    #     run_test(models=models,
    #              labels_path=args.labels_path,
    #              cfg=cfg,
    #              device=args.device)
    # else:
    #     # Run inference for each input subject
    #     run_inference(models=models,
    #                   input_path=args.input_path,
    #                   output_path=args.output_path,
    #                   cfg=cfg,
    #                   device=args.device,
    #                   return_prediction=False)
