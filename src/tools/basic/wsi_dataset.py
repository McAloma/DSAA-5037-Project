import sys, os
sys.path.append("your/path")
from PIL import Image
from torch.utils.data import Dataset


class WSIImageDataset(Dataset):
    def __init__(self, images, transform):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image
    

class WSIPathDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image
    

class WSIClassifyDataset(Dataset):
    def __init__(self, label_level="site", transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform
        
        self.data_path = f"data/TCGA_thumbnail"
        sites = os.listdir(self.data_path)

        for site in sites:
            site_path = os.path.join(self.data_path, site)
            subtypes = os.listdir(site_path)
            for subtype in subtypes:
                subtype_path = os.path.join(site_path, subtype)
                images = os.listdir(subtype_path)
                for image in images:
                    image_path = os.path.join(subtype_path, image)
                    self.image_paths.append(image_path)

                    if label_level == "site":
                        self.labels.append(site)
                    elif label_level == "subtype":
                        self.labels.append(subtype)
                    else:
                        raise ValueError("label_level must be 'site' or 'subtype'")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label 