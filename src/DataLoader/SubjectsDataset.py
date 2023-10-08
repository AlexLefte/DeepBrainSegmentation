import torch
from torch.utils.data import Dataset
import os
import nibabel as nib
from torchvision import transforms

ORIGINAL_SCAN = 't1weighted.nii.gz'
LABELS = 'labels.DKT31.manual+aseg.nii.gz'


class SubjectsDataset(Dataset):
    """
    Class used to load and process the MRI scans for network inference
    """

    def __init__(self,
                 path: str,
                 transform: transforms = None,
                 target_transform: transforms = None):
        """
        Constructor
        """
        self.path = path
        self.transform = transform
        self.target_transform = target_transform
        self.subjects = [s for s in os.listdir(path) if os.path.isdir(os.path.join(path, s))]

    def __len__(self):
        """
        Returns the length of the custom dataset.
        Must be implemented.
        """
        return len(self.subjects)

    def __getitem__(self, idx):
        """
        Returns the image data of the patient and the labels.
        Must be implemented.
        """
        subject_path = os.path.join(self.path, self.subjects[idx])
        image = nib.load(os.path.join(subject_path, ORIGINAL_SCAN)).get_fdata()
        image_tensor = torch.from_numpy(image)
        label = nib.load(os.path.join(subject_path, LABELS)).get_fdata()
        label_tensor = torch.from_numpy(label)

        if self.transform:
            image_tensor = self.transform(image)
        if self.target_transform:
            label_tensor = self.target_transform(label)

        return image_tensor, label_tensor
