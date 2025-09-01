import os
from PIL import Image
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None, label=0):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, fname)
                            for fname in os.listdir(image_dir)
                            if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.label = label

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.label

class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None, label=0, allowed_files=None):
        self.image_dir = image_dir
        self.transform = transform
        self.label = label
        
        all_files = [fname for fname in os.listdir(image_dir) if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if allowed_files is not None:
            # Filter files to exclude those in the "disallowed" list
            filtered_files = [f for f in all_files if f not in allowed_files]
        else:
            filtered_files = all_files
        
        self.image_paths = [os.path.join(image_dir, fname) for fname in filtered_files]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.label
