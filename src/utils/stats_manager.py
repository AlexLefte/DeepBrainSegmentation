import os
from torch.utils.tensorboard import SummaryWriter
import logging
from src.utils.metrics import *

LOGGER = logging.getLogger(__name__)


class StatsManager:
    """
    Class that stores training/test stats and also writes them into the SummaryWriter

    Methods
    -------
    __init__
        Constructor
    """

    def __init__(self,
                 cfg):
        """
        Constructor

        Parameters
        ----------
        num_classes: int
            The number of classes
        path: string
            The experiment's path
        """
        self.cfg = cfg
        self.num_classes = cfg['num_classes']
        self.plane_path = os.path.join(cfg['base_path'], cfg['experiments_path'].format(cfg['plane']))
        self.exp_name = cfg['exp_name']
        self.plane = cfg['plane']
        self.summary_path = os.path.join(cfg['base_path'], cfg['summary_path'].format(self.plane), self.exp_name)
        self.summary_writer = SummaryWriter(os.path.join(self.summary_path))
        self.batch_losses = []
        self.y_pred = []
        self.y_true = []
        self.batch_idx = 1
        self.epochs = cfg['epochs']
        self.results = {}

    def update_batch_stats(self,
                           preds: Tensor,
                           labels: Tensor):
        """
        Saves the loss, the learning rate, and other stats

        Parameters
        ----------
        preds:
            Predictions
        labels:
            Ground truth
        """
        # Append the predictions and labels
        self.y_pred.extend(preds.cpu().numpy())
        self.y_true.extend(labels.cpu().numpy())

    def update_epoch_stats(self,
                           mode: str,
                           loss: float,
                           learning_rate: float,
                           epoch: int):
        """
        Updates the confusion matrix, precision-recall curve and dice similarity coefficient
        """
        # Concatenate both the predictions and labels
        y_pred_concat = np.concatenate(self.y_pred)
        y_true_concat = np.concatenate(self.y_true)

        # Flatten these np.ndarray-type objects
        y_pred_flat = y_pred_concat.flatten()
        y_true_flat = y_true_concat.flatten()

        # Write the loss value
        self.summary_writer.add_scalar(f'Loss/{mode}', loss, epoch)
        self.results[f'{mode}_loss'] = loss

        # Write the accuracy and the confusion matrix
        accuracy = get_accuracy(y_pred_flat,
                                y_true_flat,
                                self.num_classes)
        self.summary_writer.add_scalar(f'Accuracy/{mode}', accuracy, epoch)
        self.results[f'{mode}_acc'] = accuracy
        # conf_matrix = self.accuracy.get_matrix()
        # self.summary_writer.add_figure(f'Confusion_matrix/{mode}', conf_matrix, epoch)

        # Write the dice similarity coefficient and its associated matrix
        dice = get_overall_dsc(y_pred_flat,
                               y_true_flat)
        self.summary_writer.add_scalar(f'DSC/{mode}', dice, epoch)
        # dice_matrix = self.dice.get_matrix()
        # self.summary_writer.add_figure(f'Dice_matrix/{mode}', dice_matrix, epoch)

        # Write the mean dsc (per class)
        # dice_per_class = get_class_dsc(y_pred_flat,
        #                                y_true_flat,
        #                                self.num_classes)
        dice_sub, dice_cort, dice_mean = get_cortical_subcortical_class_dsc(y_pred_flat,
                                                                            y_true_flat,
                                                                            self.num_classes)
        # self.summary_writer.add_scalar(f'DSC_mean_per_class/{mode}', dice_per_class, epoch)
        self.summary_writer.add_scalar(f'DSC_sub/{mode}', dice_sub, epoch)
        self.summary_writer.add_scalar(f'DSC_cort/{mode}', dice_cort, epoch)
        self.summary_writer.add_scalar(f'DSC_mean_per_class/{mode}', dice_mean, epoch)
        self.results[f'{mode}_mean_dsc'] = dice_mean
        self.results[f'{mode}_cort_dsc'] = dice_cort
        self.results[f'{mode}_sub_dsc'] = dice_sub

        # Write the average Hausdorff distance
        avg_hd_sub, avg_hd_cort, avg_hd, test = get_cort_subcort_avg_hausdorff(y_pred_flat,
                                                                               y_true_flat,
                                                                               self.num_classes)
        self.summary_writer.add_scalar(f'Avg_HD/{mode}', avg_hd, epoch)
        self.summary_writer.add_scalar(f'Avg_HD_sub/{mode}', avg_hd_sub, epoch)
        self.summary_writer.add_scalar(f'Avg_HD_cort/{mode}', avg_hd_cort, epoch)
        self.summary_writer.add_scalar(f'Avg_HD_direct/{mode}', test, epoch)

        # Write the current learning rate:
        if mode == 'train':
            self.summary_writer.add_scalar(f'Learning rate', learning_rate, epoch)

        # Log some info
        mode = str.capitalize(mode)
        LOGGER.info(f"Epoch: {epoch} | {mode} loss: {loss:.4f} | {mode} dsc: {dice_mean:.4f} | "
                    f"{mode} accuracy: {accuracy:.4f}")

        # Log the confusion matrix at the end of the training session
        if epoch == self.epochs - 1:
            dice_scores = get_class_dsc(y_pred_flat,
                                        y_true_flat,
                                        self.num_classes,
                                        return_mean=False)
            LOGGER.info(f"{mode} mode DSC results:")
            for i in range(self.num_classes):
                LOGGER.info(f"Class {i} dice score: {dice_scores[i]}")

            cf_matr = get_confusion_matrix(y_pred_flat,
                                           y_true_flat,
                                           self.num_classes)

            # Convert confusion matrix to Pandas DataFrame
            conf_matrix_df = pd.DataFrame(cf_matr, index=range(len(cf_matr)),
                                          columns=range(len(cf_matr[0])))

            # Save DataFrame to CSV file
            conf_matrix_df.to_csv(f'confusion_matrix_{mode}.csv', index=False)

            # Save the results into a xlsx file
            if mode == 'Val':
                # Save to xlsx
                self.save2csv(sheet_name='Unified_focal_loss_gamma')

        # Reset the prediction/ground truth lists
        self.y_pred = []
        self.y_true = []

        # return dice_per_class
        return dice_mean

    def update_pr_curves(self):
        # TODO
        pass

    def save2csv(self,
                 sheet_name: str):
        """
        Saves experiment results into an xlsx file
        """
        # Save some cfg settings
        to_save = ['exp_name', 'preprocessing_modality', 'data_augmentation', 'slice_thickness',
                   'batch_size', 'loss_function', 'loss_gamma', 'optimizer', 'filters', 'conv_kernel', 'lr_scheduler',
                   'lr', 'lr_step', 'lr_gamma', 'lr_restart', 'lr_mult', 'lr_min']
        cfg_dict = {key: self.cfg[key] for key in to_save}

        # Extend with the training/eval stats
        results = {**cfg_dict, **self.results}

        excel_name = self.plane + '.xlsx'  # Change the file extension to .xlsx
        excel_path = os.path.join(self.plane_path, excel_name)

        try:
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
        except FileNotFoundError:
            df = pd.DataFrame(columns=list(results.keys()))

        # Create a new DataFrame with the current results
        new_df = pd.DataFrame([results])

        # Check if the existing DataFrame is empty or has all-NA entries
        if df.empty or df.isna().all().all():
            df = new_df
        else:
            # Concatenate the existing DataFrame and the new DataFrame
            df = pd.concat([df, new_df], ignore_index=True)

        # Save the updated Excel file
        df.to_excel(excel_path, sheet_name=sheet_name, index=False)
