#Written by Devorah Rotman and Carmit Kaye


from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from torchvision import transforms


class ImageFolderDataset(Dataset):
    def __init__(self, image_dir, image_size=128):
        self.image_paths = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(image), os.path.basename(self.image_paths[idx])


def get_content_loader(content_dir, batch_size=8, image_size=128, num_workers=4):
    dataset = ImageFolderDataset(content_dir, image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    return loader


def get_style_loader(style_dir, batch_size=8, image_size=128, num_workers=4):
    dataset = ImageFolderDataset(style_dir, image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    return loader
