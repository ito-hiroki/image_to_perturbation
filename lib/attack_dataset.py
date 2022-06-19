from pathlib import Path

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class ITNAttackDataset(Dataset):
    def __init__(self, origin_path, protect_path, transform=None, is_train=True):
        self.protect_path = Path(protect_path) / "train" if is_train else "test"
        self.origin_path = Path(origin_path) / "train" if is_train else "test"
        self.is_train = is_train
        assert len(list(Path(self.protect_path).iterdir())) == len(
            list(Path(self.origin_path).iterdir())
        )
        self.length = len(list(Path(self.protect_path).iterdir()))
        self.transform = transform
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        original_image = np.array(Image.open(self.origin_path / f"{idx:0>6}.png"))
        trans_image = np.array(Image.open(self.protect_path / f"{idx:0>6}.png"))

        augmented = self.transform(image=original_image, mask=trans_image)
        original_image, trans_image = augmented["image"], augmented["mask"]

        original_image, trans_image = self.to_tensor(original_image), self.to_tensor(
            trans_image
        )
        return trans_image, original_image
