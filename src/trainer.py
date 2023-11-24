import torch
from torch.optim import lr_scheduler as lr_scheduler
from torch.optim import Optimizer as Optimizer
from models.fcnn_model import FCnnModel
from tqdm import tqdm
from time import time

from data import data_loader
from data import data_utils as du

import matplotlib.pyplot as plt


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
                 model: FCnnModel,
                 loss_fn: torch.nn.Module,
                 optim: Optimizer,
                 scheduler: lr_scheduler,
                 device: str = 'cpu'):
        """
        Constructor
        """
        # Start by setting a random seed
        torch.manual_seed(42)

        self.cfg = cfg
        self.model = model
        self.train_loader = data_loader.get_data_loader(cfg=self.cfg,
                                                        data_path=self.cfg['data_path'],
                                                        batch_size=self.cfg['batch_size'],
                                                        mode='train')
        self.eval_loader = data_loader.get_data_loader(cfg=self.cfg,
                                                       data_path=self.cfg['data_path'],
                                                       batch_size=self.cfg['batch_size'],
                                                       mode='eval')
        self.loss_fn = loss_fn
        self.optimizer = optim
        self.lr_scheduler = scheduler
        self.device = device
        self.print_stats = cfg['print_stats']
        self.epochs = cfg['epochs']

    def train_step(self):
        """
        Training step.
        """
        # Initiate the training step
        # logger.info('Training started')
        start_time = time()

        # Set up the training mode
        self.model.train()

        # Initialize the train loss and accuracy
        train_loss, train_acc = [], []
        ce_loss_list, dice_loss_list = [], []

        # Initialize the mean stats
        mean_loss, mean_acc = 0, 0

        # Loop through the data loader
        for batch_idx, batch in tqdm(enumerate(self.train_loader)):
            # Get the slices, labels and weights, then send them to the desired device
            images = batch['image'].to(self.device).float()
            labels = batch['labels'].to(self.device)
            weights = batch['weights'].to(self.device).float()
            weights_dict = batch['weights_dict'].to(self.device)[0].float()

            # Get the unique values in the label matrix
            unique_classes, counts = torch.unique(labels, return_counts=True)
            unique_classes, counts = torch.unique(weights, return_counts=True)

            # Zero the gradients before every batch
            self.optimizer.zero_grad()

            # Forward pass
            y_pred = self.model(images)

            # Compute the loss
            loss, ce_loss, dice_loss = self.loss_fn(y_predict=y_pred,
                                                    y=labels,
                                                    weights=weights,
                                                    weights_dict=weights_dict)
            # loss = self.loss_fn(y_predict=y_pred,
            #                     y=labels,
            #                     weights=weights,
            #                     weights_dict=weights_dict)

            # Backward pass
            loss.backward()

            # Optimizer step
            self.optimizer.step()

            # Append the running loss and accuracy
            train_loss.append(loss.item())
            ce_loss_list.append(ce_loss.item())
            dice_loss_list.append(dice_loss.item())
            y_pred_class = torch.argmax(y_pred, dim=1)
            d, h, w = labels.shape
            train_acc.append((y_pred_class == labels).sum().item() / (d * h * w) * 100)
            # train_acc.append((y_pred_class == labels).sum().item() / len(y_pred))

            # # Print some results from time to time:
            # if batch_idx % self.print_stats == 0 \
            #         or batch_idx == len(self.train_loader) - 1:
            #     mean_loss = sum(train_loss) / len(train_loss)
            #     print(f'Training loss on batch {batch_idx}: {mean_loss}')
            #     mean_acc = sum(train_acc) / len(train_acc)
            #     print(f'Accuracy on batch {batch_idx}: {mean_acc}')
            #     train_loss, train_acc = [], []

        # Finalize the training step
        stop_time = time()
        # logger.info(f'Training step is finished in: {stop_time - start_time} seconds.')
        print(f'Training step is finished in: {stop_time - start_time} seconds.')

        mean_loss = sum(train_loss) / len(train_loss)
        mean_ce = sum(ce_loss_list) / len(ce_loss_list)
        mean_dice = sum(dice_loss_list) / len(dice_loss_list)
        mean_acc = sum(train_acc) / len(train_acc)

        # Return the final stats:
        return mean_loss, mean_acc, mean_ce, mean_dice

    def eval_step(self):
        """
        Evaluation step
        """
        # Initiate the evaluation step
        # logger.info('Evaluation started')
        start_time = time()

        # Set up the model to eval mode
        self.model.eval()

        # Initialize the train loss and accuracy
        eval_loss, eval_acc = [], []

        # Initialize the mean stats
        mean_loss, mean_acc = 0, 0

        # Turn on the inference mode
        with ((torch.inference_mode())):
            # Loop through the data loader
            for batch_idx, batch in tqdm(enumerate(self.eval_loader)):
                # Get the slices, labels and weights, then send them to the desired device
                images = batch['image'].to(self.device).float()
                labels = batch['labels'].to(self.device).float()
                weights = batch['weights'].to(self.device).float()
                weights_dict = batch['weights_dict'].to(self.device)[1].float()

                # Zero the gradients before every batch
                self.optimizer.zero_grad()

                # Forward pass
                y_pred = self.model(images)

                # Compute the loss
                loss, ce_loss, dice_loss = self.loss_fn(y_predict=y_pred,
                                                        y=labels,
                                                        weights=weights,
                                                        weights_dict=weights_dict)

                # Backward pass
                loss.backward()

                # Optimizer step
                self.optimizer.step()

                # Append the running loss and accuracy
                eval_loss.append(loss.item())
                y_pred_class = torch.argmax(y_pred, dim=1)
                h, w = labels.shape
                eval_acc.append((y_pred_class == labels).sum().item() / (h * w) * 100)

                # Print some results from time to time:
                if batch_idx % self.print_stats == 0 \
                        or batch_idx == len(self.train_loader) - 1:
                    # Compute the mean loss
                    mean_loss = sum(eval_loss) / len(eval_loss)
                    print(f'Evaluation loss on batch {batch_idx}: {mean_loss}')

                    # Compute the mean accuracy
                    mean_acc = sum(eval_acc) / len(eval_acc)
                    print(f'Accuracy on batch {batch_idx}: {mean_acc}')
                    eval_loss, eval_acc = [], []

        # Finalize the training step
        stop_time = time()
        # logger.info(f'Evaluation step is finished in: {stop_time - start_time} seconds.')
        print(f'Evaluation step is finished in: {stop_time - start_time} seconds.')

        # Return the stats
        return mean_loss, mean_acc

    def train(self):
        # Transfer to device
        self.model = self.model.to(self.device)

        # Create empty results dictionary
        results = {"train_loss": [],
                   "train_acc": [],
                   "test_loss": [],
                   "test_acc": []}

        ce_loss_list, dice_loss_list = [], []

        # Start training
        start_time = time()

        # Loop through training and testing steps for a number of epochs
        for epoch in tqdm(range(self.epochs)):
            train_loss, train_acc, ce_loss, dice_loss = self.train_step()

            #  test_loss, test_acc = self.eval_step()
            test_loss, test_acc = 0, 0

            # Print out what's happening
            # print(f"Epoch: {epoch} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | "
            #       f"Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")
            # print(f"Epoch: {epoch} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")
            print(f"Epoch: {epoch} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | CE Loss: {ce_loss:.4f}"
                  f" | Dice Loss: {dice_loss:.4f}")

            # Update results dictionary
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)
            ce_loss_list.append(ce_loss)
            dice_loss_list.append(dice_loss)

        # Stop training
        end_time = time()
        print(f"Total training time: {end_time - start_time:.3f} seconds")

        # plt.figure(figsize=(15, 7))
        # plt.plot(range(self.epochs), results["train_loss"])
        # # plt.plot(epochs, test_loss, label="test_loss")
        # plt.title("Loss")
        # plt.xlabel("Epochs")
        # plt.show()

        # Plot the loss/acc curves
        du.plot_loss_curves(results)

        #
        # plot CE and Dice losses
        # Figure out how many epochs there were
        epochs = range(len(ce_loss_list))

        # Setup a plot
        plt.figure(figsize=(15, 7))

        # Plot the loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, ce_loss_list)
        plt.title("CE Loss")
        plt.xlabel("Epochs")

        # Plot the accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, dice_loss_list)
        plt.title("Dice Loss")
        plt.xlabel("Epochs")
        plt.legend()
        plt.show()
