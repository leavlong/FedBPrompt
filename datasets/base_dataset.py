import os
import torch
from torch.utils.data import Dataset
from PIL import Image


class BaseDataset(Dataset):
    """Base dataset class for BAPM.

    Override `load_annotations` and `__getitem__` in subclasses.
    """

    def __init__(self, root, split='train', transform=None):
        """
        Args:
            root: path to the dataset root directory
            split: dataset split, one of 'train', 'val', 'test'
            transform: optional transform applied to images
        """
        self.root = root
        self.split = split
        self.transform = transform
        self.samples = self.load_annotations()

    def load_annotations(self):
        """Load sample annotations. Override in subclasses.

        Returns:
            list of dicts, each with keys 'query', 'key', 'label'
        """
        raise NotImplementedError

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path):
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_q = self._load_image(sample['query'])
        img_k = self._load_image(sample['key'])
        label = torch.tensor(sample['label'], dtype=torch.long)
        return img_q, img_k, label
