import os
from torchvision import datasets, transforms
from PIL import Image

# Grayscale-aware image loader
def pil_loader_grayscale(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')

# Evaluation transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Dataset fetcher
def get_datasets(data_dir=r"C:\Users\USER\Downloads\Pneumonia dataset\chest_xray"):
    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'train'),
        transform=transform,
        loader=pil_loader_grayscale
    )
    val_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'val'),
        transform=transform,
        loader=pil_loader_grayscale
    )
    test_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'test'),
        transform=transform,
        loader=pil_loader_grayscale
    )
    return train_dataset, val_dataset, test_dataset
