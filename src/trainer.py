import torch.utils.data
from torch.optim import lr_scheduler as lr_scheduler
from torch.optim import Optimizer as Optimizer
from src.models.fcnn_model import FCnnModel
from tqdm import tqdm
from time import time
from src.utils.stats_manager import StatsManager
from src.utils.checkpoint import *

import logging

LOGGER = logging.getLogger(__name__)


class Trainer:
    """
    Trainer class

    Methods
    -------
    __init__
        Constructor
    train_step
        Performs the training step
    eval_step
        Performs the evaluation step
    train
        Defines the training loop
    """

    def __init__(self,
                 cfg: dict,
                 train_loader: torch.utils.data.DataLoader,
                 val_loader: torch.utils.data.DataLoader,
                 model: FCnnModel,
                 loss_fn: torch.nn.Module,
                 optim: Optimizer,
                 lr_sch: lr_scheduler,
                 stats_manager: StatsManager,
                 device: str = 'cpu'):
        """
        Constructor
        """
        # Start by setting a random seed
        torch.manual_seed(42)

        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optim
        self.lr_scheduler = lr_sch
        self.device = device
        self.print_stats = cfg['print_stats']
        self.epochs = cfg['epochs']
        self.stats = stats_manager
        self.checkpoint_path = cfg['checkpoint_path']

    def train_step(self,
                   epoch: int):
        """
        Training step.
        """
        # Initialize the training loss
        train_loss = 0

        # Set up the training mode
        self.model.train()

        # Loop through the data loader
        for batch_idx, batch in tqdm(enumerate(self.train_loader)):
            # Get the slices, labels and weights, then send them to the desired device
            images = batch['image'].to(self.device).float()
            labels = batch['labels'].to(self.device)
            weights = batch['weights'].to(self.device).float()
            weights_list = batch['weights_list'].to(self.device).float()

            # Zero the gradients before every batch
            self.optimizer.zero_grad()

            # Forward pass
            y_pred = self.model(images)

            # Compute the loss
            loss = self.loss_fn(y_pred=y_pred,
                                y_true=labels,
                                weights=weights,
                                weights_list=weights_list)

            # Update the running loss:
            train_loss += loss.item()

            # Backward pass
            loss.backward()

            # Optimizer step
            self.optimizer.step()

            # Compute argmax over the prediction tensor
            y_pred_class = torch.argmax(y_pred, dim=1)

            # Write stats per batch
            self.stats.update_batch_stats(preds=y_pred_class,
                                          labels=labels)

            # # Log some results from time to time:
            # if batch_idx % self.print_stats == 0:
            #     mean_loss = sum(train_loss) / len(train_loss)
            #     print(f'Training loss on batch {batch_idx}: {mean_loss}')
            #     mean_acc = sum(train_acc) / len(train_acc)
            #     print(f'Accuracy on batch {batch_idx}: {mean_acc}')
            #     train_loss, train_acc = [], []

        # Compute the mean loss / epoch
        train_loss /= len(self.train_loader)

        # Update stats per epoch
        if self.lr_scheduler is None:
            self.stats.update_epoch_stats(mode='train',
                                          loss=train_loss,
                                          learning_rate=self.cfg['lr'],
                                          epoch=epoch)
        else:
            self.stats.update_epoch_stats(mode='train',
                                          loss=train_loss,
                                          learning_rate=self.lr_scheduler.get_last_lr()[0],
                                          epoch=epoch)

    def eval_step(self,
                  epoch: int):
        """
        Evaluation step
        """
        # Set up a loss list
        eval_loss = 0

        # Set up the model to eval mode
        self.model.eval()

        # Turn on the inference mode
        with torch.inference_mode():
            # Loop through the data loader
            for batch_idx, batch in tqdm(enumerate(self.val_loader)):
                # Get the slices, labels and weights, then send them to the desired device
                images = batch['image'].to(self.device).float()
                labels = batch['labels'].to(self.device)
                weights = batch['weights'].to(self.device).float()
                weights_list = batch['weights_list'].to(self.device).float()

                # Forward pass
                y_pred = self.model(images)

                # Compute the loss
                loss = self.loss_fn(y_pred=y_pred,
                                    y_true=labels,
                                    weights=weights,
                                    weights_list=weights_list)

                # Add the running loss
                eval_loss += loss.item()

                # Apply argmax over the prediction tensor
                y_pred_class = torch.argmax(y_pred, dim=1)

                # Write stats per batch
                self.stats.update_batch_stats(preds=y_pred_class,
                                              labels=labels)

                # # Log some results from time to time:
                # if batch_idx % self.print_stats == 0 \
                #         or batch_idx == len(self.train_loader) - 1:
                #     # Compute the mean loss
                #     mean_loss = sum(eval_loss) / len(eval_loss)
                #     print(f'Evaluation loss on batch {batch_idx}: {mean_loss}')
                #
                #     # Compute the mean accuracy
                #     mean_acc = sum(eval_acc) / len(eval_acc)
                #     print(f'Accuracy on batch {batch_idx}: {mean_acc}')
                #     eval_loss, eval_acc = [], []

        # Compute the mean loss / epoch
        eval_loss /= len(self.val_loader)

        # Update stats per epoch
        if self.lr_scheduler is None:
            dsc = self.stats.update_epoch_stats(mode='val',
                                                loss=eval_loss,
                                                learning_rate=self.cfg['lr'],
                                                epoch=epoch)
        else:
            dsc = self.stats.update_epoch_stats(mode='val',
                                                loss=eval_loss,
                                                learning_rate=self.lr_scheduler.get_last_lr()[0],
                                                epoch=epoch)
        return dsc

    def train(self):
        # try:
        #     for batch_idx, batch in tqdm(enumerate(self.train_loader)):
        #         continue
        # except Exception as e:
        #     print(f'Exception in for -> {e}')
        #
        # print('Finished iterating through the data loader.')
        # return

        # Initialize the best dice score
        best_dsc = 0.0
        dsc = 0.0

        # Transfer to device
        self.model = self.model.to(self.device)

        # Start training
        LOGGER.info('==== Started training ====')
        start_time = time()

        # Loop through training and testing steps for a number of epochs
        for epoch in tqdm(range(self.epochs)):
            # Perform the training step
            self.train_step(epoch)

            # Perform the evaluation step
            dsc = self.eval_step(epoch)

            # Compare the current dsc with the best dsc
            if dsc > best_dsc:
                best_dsc = dsc
                LOGGER.info(
                    f"New best checkpoint at epoch {epoch + 1} | DSC: {dsc}\nSaving new best model."
                )
                save_checkpoint(path=self.checkpoint_path,
                                epoch=epoch + 1,
                                score=dsc,
                                model=self.model,
                                optimizer=self.optimizer,
                                scheduler=self.lr_scheduler,
                                is_best=True)

            # Update the learning rate
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # Save the last state of the network
        save_checkpoint(path=self.checkpoint_path,
                        epoch=self.cfg['epochs'],
                        score=dsc,
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.lr_scheduler,
                        is_latest=True)

        # Stop training
        end_time = time()
        LOGGER.info(f"==== Stopped training. Total training time: {end_time - start_time:.3f} seconds ===="
                    f"\n===========================================")
