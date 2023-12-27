import torch
import os


def save_checkpoint(path: str,
                    epoch: int = None,
                    score: float = None,
                    model: torch.nn.Module = None,
                    optimizer: torch.optim.Optimizer = None,
                    scheduler=None,
                    is_best: bool = False,
                    is_latest: bool = False,
                    ):
    """
    Stores/loads checkpoints (model states).
    """
    # Create the directory if it does not exist
    if not os.path.exists(path):
        os.makedirs(path)

    # Create the checkpoint dictionary
    checkpoint = {
        'epoch': epoch,
        'score': score,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict()
    }

    if scheduler is not None:
        checkpoint['scheduler_state'] = scheduler.state_dict()

    if is_best:
        path += '/best.pkl'
        if os.path.exists(path):
            os.remove(path)
    elif is_latest:
        path += '/latest.pkl'
    else:
        path += f'/epoch_{epoch}.pkl'

    torch.save(checkpoint, path)


